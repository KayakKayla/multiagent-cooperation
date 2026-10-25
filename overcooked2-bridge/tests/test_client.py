import importlib.util
import json
import socket
import tempfile
import threading
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('bridge_client', Path(__file__).parents[1] / 'client.py')
bridge = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)


class Server:
    def __init__(self, handler):
        self.listener = socket.socket()
        self.listener.bind(('127.0.0.1', 0))
        self.listener.listen(1)
        self.port = self.listener.getsockname()[1]
        self.error = None

        def run():
            try:
                with self.listener.accept()[0] as peer:
                    peer.settimeout(3)
                    handler(peer)
            except Exception as exc:
                self.error = exc
            finally:
                self.listener.close()

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def finish(self):
        self.thread.join(4)
        if self.thread.is_alive():
            raise AssertionError('Test server failed to finish')
        if self.error:
            raise self.error


class ClientTests(unittest.TestCase):
    def test_resuming_report_preserves_measured_results(self):
        with tempfile.TemporaryDirectory() as directory:
            first = bridge.Report(directory)
            first.metadata['assembly_sha256'] = 'abc'
            first.result('move', True, displacement=1.5)
            second = bridge.Report(directory)
            second.result('pickup', True)
            summary = json.loads((Path(directory) / 'summary.json').read_text())
            self.assertEqual([r['test'] for r in summary['results']], ['move', 'pickup'])
            self.assertEqual(summary['metadata']['assembly_sha256'], 'abc')

    def test_split_response_and_provenance_report(self):
        def handler(peer):
            request = json.loads(peer.makefile('rb').readline())
            reply = json.dumps(dict(ok=True, request_id=request['request_id'],
                                    plugin_version='test', assembly_sha256='abc', state={'chefs': []})).encode() + b'\n'
            peer.sendall(reply[:12])
            peer.sendall(reply[12:])

        server = Server(handler)
        with tempfile.TemporaryDirectory() as directory:
            report = bridge.Report(directory)
            with bridge.Client(server.port, report) as client:
                self.assertEqual(client.request('state')['state']['chefs'], [])
            summary = json.loads((Path(directory) / 'summary.json').read_text())
            self.assertEqual(summary['metadata']['assembly_sha256'], 'abc')
            events = [json.loads(line) for line in (Path(directory) / 'events.jsonl').read_text().splitlines()]
            self.assertEqual(events[-1]['request']['op'], 'state')
            self.assertIn('sent_utc', events[-1])
        server.finish()

    def test_pickup_error_is_not_retried(self):
        received = []

        def handler(peer):
            with peer.makefile('rb') as reader:
                received.append(json.loads(reader.readline()))
                peer.sendall(b'{"ok":false,"error":"hands_not_empty","message":"empty hands required"}\n')
                self.assertEqual(reader.readline(), b'')

        server = Server(handler)
        with bridge.Client(server.port) as client:
            with self.assertRaises(bridge.BridgeError) as error:
                client.request('pickup', chef_id=42, scene_token='s')
            self.assertEqual(error.exception.response['error'], 'hands_not_empty')
        server.finish()
        self.assertEqual(len(received), 1)

    def test_request_id_mismatch(self):
        def handler(peer):
            peer.makefile('rb').readline()
            peer.sendall(b'{"ok":true,"request_id":"other"}\n')

        server = Server(handler)
        with bridge.Client(server.port) as client:
            with self.assertRaisesRegex(RuntimeError, 'request_id mismatch'):
                client.request('state')
        server.finish()

    def test_missing_nested_state_is_explicit_error(self):
        def handler(peer):
            request = json.loads(peer.makefile('rb').readline())
            peer.sendall(json.dumps(dict(ok=True, request_id=request['request_id'])).encode() + b'\n')

        server = Server(handler)
        with bridge.Client(server.port) as client:
            with self.assertRaisesRegex(RuntimeError, 'missing nested state'):
                client.request('state')
        server.finish()

    def test_partial_reply_disconnect_is_failure(self):
        def handler(peer):
            peer.makefile('rb').readline()
            peer.sendall(b'{"ok":true')

        server = Server(handler)
        with bridge.Client(server.port) as client:
            with self.assertRaisesRegex(RuntimeError, 'no command was retried'):
                client.request('pickup', chef_id=42, scene_token='s')
        server.finish()


if __name__ == '__main__':
    unittest.main()
