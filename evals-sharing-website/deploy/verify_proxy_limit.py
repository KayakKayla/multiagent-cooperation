"""Check the private nginx body limit before starting the real application.

Run with the website venv: .venv/bin/python deploy/verify_proxy_limit.py
nginx must already listen on 127.0.0.1:8001; port 8000 must be free.
This temporary upstream consumes bytes without saving them or importing the app.
"""

from __future__ import annotations

from concurrent.futures import Future
import http.client
import socket
import sys
import threading
import time


LIMIT = 20 * 1024 * 1024
HOST = "127.0.0.1"
UPSTREAM_PORT = 8000
PROXY_PORT = 8001
BLOCK = b"x" * (64 * 1024)
IO_TIMEOUT = 10
REQUEST_TIMEOUT = 30
START_TIMEOUT = 5
STOP_TIMEOUT = 6


async def consume(scope, receive, send):
    """Return the byte count only after the complete request has arrived."""
    total = 0
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            return
        total += len(message.get("body", b""))
        if total > LIMIT + 1:
            # Bound even unsolicited local requests during this short check.
            await send({"type": "http.response.start", "status": 400, "headers": []})
            await send({"type": "http.response.body", "body": b"Test body too large"})
            return
        if not message.get("more_body", False):
            break
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": f"received={total}\n".encode()})


def check_request(size: int, chunked: bool) -> None:
    """Send bounded blocks while independently reading an eager rejection."""
    result = Future()
    response_started = threading.Event()
    label = "chunked" if chunked else "Content-Length"
    with socket.create_connection((HOST, PROXY_PORT), timeout=IO_TIMEOUT) as connection:
        connection.settimeout(IO_TIMEOUT)
        framing = "Transfer-Encoding: chunked" if chunked else f"Content-Length: {size}"
        connection.sendall((
            "POST /__evals_proxy_limit_check HTTP/1.1\r\n"
            "Host: rowan.gaertner.ca\r\n"
            "Connection: close\r\n"
            "Content-Type: application/octet-stream\r\n"
            f"{framing}\r\n\r\n"
        ).encode("ascii"))

        def read_response():
            try:
                with http.client.HTTPResponse(connection, method="POST") as response:
                    response.begin()
                    response_started.set()
                    # Never collect arbitrary HTML or an unbounded response.
                    body = response.read(1025)
                    if len(body) > 1024:
                        raise RuntimeError("proxy returned an unexpectedly large response")
                    result.set_result((response.status,
                                       response.getheader("X-Evals-Body-Limit"), body))
            except Exception as error:
                result.set_exception(error)
            finally:
                response_started.set()

        reader = threading.Thread(target=read_response, name="proxy-response", daemon=True)
        reader.start()
        sent = 0
        complete = False
        write_error = None
        deadline = time.monotonic() + REQUEST_TIMEOUT
        try:
            while sent < size and not response_started.is_set():
                if time.monotonic() >= deadline:
                    raise TimeoutError("upload exceeded the request time limit")
                block = BLOCK[:min(len(BLOCK), size - sent)]
                if chunked:
                    connection.sendall(f"{len(block):x}\r\n".encode() + block + b"\r\n")
                else:
                    connection.sendall(block)
                sent += len(block)
            if sent == size:
                if chunked:
                    connection.sendall(b"0\r\n\r\n")
                complete = True
        except OSError as error:
            # nginx can return 413 and close while sendall is still writing.
            # Read the HTTP result before deciding whether this was a failure.
            write_error = error
        finally:
            reader.join(IO_TIMEOUT + 1)
            if reader.is_alive():
                try:
                    connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                reader.join(1)
            if reader.is_alive():
                raise RuntimeError("response reader did not stop")

        try:
            status, cap, body = result.result()
        except Exception as error:
            raise RuntimeError(f"{label} {size}: no complete HTTP response ({error})") from error
        if size == LIMIT:
            if status != 200 or body != f"received={size}\n".encode() or not complete:
                raise RuntimeError(f"{label} {size}: expected complete 200 byte-count response; got {status}")
            if write_error is not None:
                raise RuntimeError(f"{label} {size}: upload failed ({write_error})")
        elif status != 413 or cap != str(LIMIT):
            raise RuntimeError(f"{label} {size}: expected nginx 413 with X-Evals-Body-Limit={LIMIT}; got {status}")
    print(f"PASS {label}: {size} bytes -> {status}", flush=True)


def run_checks() -> None:
    try:
        import uvicorn
    except ImportError as error:
        raise RuntimeError("use the website venv with its committed dependencies installed") from error

    # Bind ourselves before starting Uvicorn: never replace or stop a listener.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        try:
            listener.bind((HOST, UPSTREAM_PORT))
        except OSError as error:
            raise RuntimeError("cannot reserve 127.0.0.1:8000; stop this check and inspect the existing listener") from error
        listener.listen(16)
        config = uvicorn.Config(consume, host=HOST, port=UPSTREAM_PORT,
                                loop="asyncio", http="h11", lifespan="off",
                                access_log=False, log_config=None, log_level="critical",
                                limit_concurrency=4, timeout_keep_alive=1,
                                timeout_graceful_shutdown=2)
        server = uvicorn.Server(config)
        server_error = []

        def serve():
            try:
                server.run(sockets=[listener])
            except BaseException as error:
                server_error.append(error)

        worker = threading.Thread(target=serve, name="temporary-upstream", daemon=True)
        worker.start()
        try:
            deadline = time.monotonic() + START_TIMEOUT
            while not server.started:
                if not worker.is_alive() or time.monotonic() >= deadline:
                    detail = str(server_error[0]) if server_error else "startup timed out"
                    raise RuntimeError(f"temporary loopback upstream did not start ({detail})")
                time.sleep(0.02)
            for chunked in (False, True):
                for size in (LIMIT, LIMIT + 1):
                    check_request(size, chunked)
        finally:
            server.should_exit = True
            worker.join(STOP_TIMEOUT)
            if worker.is_alive():
                server.force_exit = True
                worker.join(2)
            if worker.is_alive():
                raise RuntimeError("temporary upstream did not stop within its shutdown limit")


def main() -> int:
    try:
        run_checks()
    except (Exception, KeyboardInterrupt) as error:
        print(f"FAIL: {error or 'interrupted'}", file=sys.stderr)
        return 1
    print("All four private proxy body-limit checks passed; temporary upstream stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
