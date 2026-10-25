#!/usr/bin/env python3
"""Read state and run short, manually prepared checks in the Steam game."""
import argparse
import datetime as dt
import json
import math
import socket
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


class BridgeError(RuntimeError):
    def __init__(self, response):
        self.response = response
        super().__init__(response.get('error', 'bridge_error') + ': ' + response.get('message', ''))


class Report:
    def __init__(self, directory=None):
        self.directory = Path(directory) if directory else ROOT / 'outputs' / (
            'test-' + dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
        self.directory.mkdir(parents=True, exist_ok=True)
        previous = self.directory / 'summary.json'
        existing = json.loads(previous.read_text()) if previous.exists() else {}
        self.metadata = existing.get('metadata', {})
        self.results = existing.get('results', [])
        self.append({'event': 'session', 'started_utc': utc()})

    def append(self, entry):
        with (self.directory / 'events.jsonl').open('a') as f:
            f.write(json.dumps(dict(recorded_utc=utc(), **entry), allow_nan=False) + '\n')

    def result(self, name, passed, **details):
        entry = dict(test=name, passed=passed, **details)
        self.results.append(entry)
        self.append({'event': 'result', **entry})
        self.save()

    def save(self):
        summary = dict(metadata=self.metadata, results=self.results,
                       updated_utc=utc(), all_recorded_checks_passed=bool(self.results) and all(r['passed'] for r in self.results))
        (self.directory / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')


class Client:
    def __init__(self, port=8765, report=None):
        self.socket = socket.create_connection(('127.0.0.1', port), timeout=3)
        self.reader = self.socket.makefile('rb')
        self.report = report

    def request(self, op, **fields):
        request = dict(op=op, request_id=uuid.uuid4().hex, **fields)
        start = utc()
        self.socket.sendall(json.dumps(request, allow_nan=False).encode() + b'\n')
        line = self.reader.readline(65537)
        if not line or len(line) > 65536 or not line.endswith(b'\n'):
            raise RuntimeError('Connection closed or invalid response framing; no command was retried')
        response = json.loads(line)
        if self.report:
            self.report.metadata.update({k: response[k] for k in ('plugin_version', 'assembly_sha256') if k in response})
            self.report.append(dict(event='exchange', sent_utc=start, request=request, response=response))
            self.report.save()
        if not response.get('ok'):
            raise BridgeError(response)
        if response.get('request_id') != request['request_id']:
            raise RuntimeError('Response request_id mismatch')
        if not isinstance(response.get('state'), dict):
            raise RuntimeError('Bridge response is missing nested state; check the plugin version and report')
        return response

    def close(self):
        # shutdown is needed because the buffered reader also owns a socket handle.
        try:
            self.socket.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.reader.close()
        self.socket.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def chef(state, chef_id):
    for entry in state['chefs']:
        if entry['chef_id'] == chef_id:
            return entry
    raise RuntimeError('Selected chef is unavailable; read state and select again')


def distance(a, b):
    return math.sqrt(sum((a[axis] - b[axis]) ** 2 for axis in ('x', 'y', 'z')))


def show(state):
    print('Scene:', state['scene'], '| token:', state['scene_token'])
    print('Paused:', state['paused'], '| in round:', state['in_round'],
          '| local single player:', state['local_single_player'])
    for entry in state['chefs']:
        print('Chef', entry['chef_id'], entry['name'], '| selected:', entry['selected'],
              '| ready:', entry['can_control'], '| position:', entry['position'],
              '| held:', entry['held'])


def countdown(seconds):
    if seconds:
        print('Focus the game and resume the level. Sending in', seconds, 'seconds.', flush=True)
        time.sleep(seconds)


def action(client, op, chef_id, scene_token, report, x=1, y=0, duration_ms=250):
    fields = dict(chef_id=chef_id, scene_token=scene_token)
    if op == 'move':
        fields.update(x=x, y=y, duration_ms=duration_ms)
    try:
        accepted = client.request(op, **fields)
        time.sleep((duration_ms if op == 'move' else 150) / 1000 + 0.15)
        after = client.request('state')['state']
    finally:
        try:
            client.request('release')
        except (OSError, RuntimeError):
            pass  # The server also releases input when this connection closes.
    before = accepted['before']
    if after['scene_token'] != before['scene_token']:
        report.result(op, False, reason='scene_changed', before=before, after=after)
        return False
    first, last = chef(before, chef_id), chef(after, chef_id)
    if op == 'move':
        delta = distance(first['position'], last['position'])
        # A further idle sample distinguishes displacement from continued input.
        time.sleep(0.2)
        stopped = client.request('state')['state']
        drift = distance(last['position'], chef(stopped, chef_id)['position'])
        passed = delta > 0.01 and drift < 0.05 and not after['input_active']
        report.result(op, passed, displacement=delta, idle_drift=drift, before=before, after=after, stopped=stopped)
        print('Movement:', 'PASS' if passed else 'FAIL', '| displacement:', round(delta, 4), '| idle drift:', round(drift, 4))
    else:
        passed = (first['held']['known'] and not first['held']['occupied']
                  and last['held']['known'] and last['held']['occupied'] and not after['input_active'])
        report.result(op, passed, before=before, after=after)
        print('Pickup:', 'PASS' if passed else 'FAIL', '| held object:', last['held'])
    return passed


def select(client):
    state = client.request('state')['state']
    show(state)
    selected = int(input('Enter the chef_id to test: ').strip())
    chef(state, selected)
    return selected


def prepared_state(client, selected, delay):
    countdown(delay)
    state = client.request('state')['state']
    if not chef(state, selected)['can_control']:
        show(state)
        raise RuntimeError('Chef is not ready. Select this chef in the game and resume; no action was sent.')
    return state


def verify(args, report):
    print('This is a manual test wizard. Each game input is at most 500 ms. No actions are retried.')
    with Client(args.port, report) as client:
        selected = select(client)
        first = client.request('state')['state']
        input('Move that chef manually, then return here and press Enter to sample the new position. ')
        last = client.request('state')['state']
        read_passed = (first['scene_token'] == last['scene_token'] and
                       distance(chef(first, selected)['position'], chef(last, selected)['position']) > 0.01)
        report.result('manual_state_read', read_passed, before=first, after=last)
        if not read_passed:
            raise RuntimeError('No manual position change was observed. Resolve state reading first.')
        for repetition in range(1, 4):
            input('Movement %d/3: put this chef on clear, stationary floor, resume, then press Enter here. ' % repetition)
            state = prepared_state(client, selected, args.delay)
            if not action(client, 'move', selected, state['scene_token'], report, args.x, args.y):
                raise RuntimeError('Movement check failed; inspect the report before another attempt.')
            input('Pickup %d/3: make the chef empty-handed beside an ingredient dispenser, facing it; resume and press Enter here. ' % repetition)
            state = prepared_state(client, selected, args.delay)
            if not action(client, 'pickup', selected, state['scene_token'], report):
                raise RuntimeError('Pickup check failed; no pickup was retried.')
            visual = input('Does the chef visibly hold the expected ingredient? [y/N] ').strip().lower() == 'y'
            report.result('ingredient_visual_%d' % repetition, visual)
            if not visual:
                raise RuntimeError('Visual ingredient check failed.')
        input('Disconnect check: place the chef on clear floor, resume, then press Enter here. ')
        state = prepared_state(client, selected, args.delay)
        client.request('move', chef_id=selected, scene_token=state['scene_token'], x=args.x, y=args.y, duration_ms=500)
        time.sleep(0.05)
        client.close()
        time.sleep(0.25)
    with Client(args.port, report) as client:
        after = client.request('state')['state']
        # Check before the original 500 ms deadline: this exercises disconnect release.
        elapsed = after['monotonic_seconds'] - state['monotonic_seconds']
        timely = elapsed < 0.5
        report.result('disconnect_release', not after['input_active'] and timely, after=after, elapsed_seconds=elapsed)
        if after['input_active'] or not timely:
            raise RuntimeError('Disconnect did not release input promptly.')
        old_token = after['scene_token']
        input('Scene check: leave this level and load a level again, then press Enter here. ')
        fresh = client.request('state')['state']
        if fresh['scene_token'] == old_token:
            raise RuntimeError('No scene change observed; scene check remains incomplete.')
        try:
            client.request('move', chef_id=selected, scene_token=old_token, x=1, y=0, duration_ms=1)
        except BridgeError as exc:
            passed = exc.response.get('error') == 'stale_scene'
        else:
            client.request('release')
            passed = False
        report.result('stale_scene_rejected', passed)
        if not passed:
            raise RuntimeError('Stale scene command was not rejected as expected.')
    report.result('wizard_completed', True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--report-dir', type=Path)
    commands = parser.add_subparsers(dest='op', required=True)
    commands.add_parser('state')
    commands.add_parser('release')
    for op in ('move', 'pickup'):
        p = commands.add_parser(op)
        p.add_argument('--chef', type=int, required=True)
        p.add_argument('--scene-token', required=True)
        p.add_argument('--delay', type=float, default=3)
        if op == 'move':
            p.add_argument('--x', type=float, default=1)
            p.add_argument('--y', type=float, default=0)
            p.add_argument('--duration-ms', type=int, default=250)
    p = commands.add_parser('verify')
    p.add_argument('--delay', type=float, default=3)
    p.add_argument('--x', type=float, default=1)
    p.add_argument('--y', type=float, default=0)
    args = parser.parse_args()
    if args.op in ('move', 'verify') and (not math.isfinite(args.x) or not math.isfinite(args.y)
                                        or not 0 < args.x ** 2 + args.y ** 2 <= 1):
        parser.error('Movement vector must have length greater than 0 and at most 1')
    if args.op == 'move' and not 1 <= args.duration_ms <= 500:
        parser.error('--duration-ms must be between 1 and 500')
    if hasattr(args, 'delay') and (not math.isfinite(args.delay) or not 0 <= args.delay <= 60):
        parser.error('--delay must be between 0 and 60 seconds')
    report = Report(args.report_dir)
    try:
        if args.op == 'verify':
            verify(args, report)
        else:
            countdown(getattr(args, 'delay', 0))
            with Client(args.port, report) as client:
                if args.op == 'state':
                    show(client.request('state')['state'])
                elif args.op == 'release':
                    client.request('release')
                    print('Input released.')
                else:
                    passed = action(client, args.op, args.chef, args.scene_token, report,
                                    getattr(args, 'x', 1), getattr(args, 'y', 0), getattr(args, 'duration_ms', 250))
                    if not passed:
                        return 2
    except (OSError, RuntimeError, ValueError, EOFError, KeyboardInterrupt) as exc:
        report.append({'event': 'error', 'message': str(exc)})
        report.result('command_completed', False, reason=str(exc))
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        report.save()
        print('Report:', report.directory)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
