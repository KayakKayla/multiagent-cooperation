#!/usr/bin/env python3
"""Record read-only bridge samples and analyze their timing evidence offline."""
import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from client import Client, ROOT, utc


def clock(state, layer):
    values = state.get('layer_game_seconds', [])
    return values[layer] if isinstance(layer, int) and 0 <= layer < len(values) else None


def quantiles(values):
    values = sorted(values)
    if not values:
        return None
    return {'count': len(values), 'min': values[0], 'median': statistics.median(values),
            'p95': values[min(len(values) - 1, math.ceil(len(values) * .95) - 1)], 'max': values[-1]}


class Evidence:
    """Conservative events: component counters are evidence, object loss is not completion."""
    def __init__(self, interval=.1):
        self.interval = interval
        self.previous = None
        self.samples = 0
        self.intervals, self.latencies, self.capture_ms = [], [], []
        self.events, self.rates = [], []
        self.warnings = Counter()
        self.scenes, self.provenance, self.stations, self.return_configs = {}, {}, {}, {}
        self.chops = {}

    def add(self, sample):
        response = sample['response']
        state = response['state']
        token = state['scene_token']
        kitchen = state.get('kitchen', {})
        previous = self.previous
        self.samples += 1
        self.latencies.append(sample['received_monotonic'] - sample['sent_monotonic'])
        if state.get('capture_ms') is not None:
            self.capture_ms.append(state['capture_ms'])
        self.provenance[token] = {k: response.get(k) for k in ('plugin_version', 'assembly_sha256')}
        self.scenes[token] = state['scene']
        self.warnings.update(kitchen.get('warnings', []))
        self.stations[token] = kitchen.get('stations', [])
        self.return_configs[token] = (kitchen.get('plate_return') or {}).get('configured_seconds')
        start = len(self.events)
        if previous:
            host_delta = sample['sent_monotonic'] - previous['sent_monotonic']
            self.intervals.append(host_delta)
            old = previous['response']['state']
            if old['scene_token'] != token:
                self.events.append({'event': 'scene_changed', 'from': old['scene_token'], 'to': token})
                self.chops.clear()
            else:
                self.compare(old, state, host_delta)
        else:
            self.begin_chops(state)
        for event in self.events[start:]:
            event.update(sample=sample['sequence'], recorded_utc=sample['received_utc'], scene_token=token)
        self.previous = sample
        return self.events[start:]

    def begin_chops(self, state):
        items = {i['item_id']: i for i in state['kitchen']['items']}
        for station in state['kitchen']['stations']:
            item = items.get(station.get('item_id'))
            if item and item.get('chopping'):
                key = (station['station_id'], item['item_id'])
                self.chops.setdefault(key, {'lower': 0., 'upper': 0., 'interrupted': False,
                    'multiple_workers': False, 'sampling_gap': False,
                    'left_censored': bool((station.get('work') or {}).get('interactor_count', 0)) or
                                     item['chopping'].get('completion_fraction') != 0})

    def compare(self, old, new, host_delta):
        old_items = {i['item_id']: i for i in old['kitchen']['items']}
        new_items = {i['item_id']: i for i in new['kitchen']['items']}
        new_stations = {s['station_id']: s for s in new['kitchen']['stations']}
        gap = host_delta > max(.3, 3 * self.interval)
        paused = old['paused'] or new['paused']
        pending_a = (old['kitchen'].get('plate_return') or {}).get('pending', [])
        pending_b = (new['kitchen'].get('plate_return') or {}).get('pending', [])
        if len(pending_a) != len(pending_b):
            self.events.append({'event': 'plate_return_queue_changed', 'before': pending_a, 'after': pending_b,
                                'meaning': 'Queue change; inspect physical plate counts to confirm arrival'})
        if len(pending_a) == len(pending_b) == 1 and pending_a[0]['station_id'] == pending_b[0]['station_id']:
            # A countdown runs in the opposite direction to cooking/washing.
            self.rate('plate_return', pending_b[0]['station_id'], pending_b[0].get('remaining_seconds'),
                      pending_a[0].get('remaining_seconds'), old, new, 0, gap or paused)
        if old['paused'] != new['paused']:
            self.events.append({'event': 'pause_changed', 'paused': new['paused']})
        for item_id in new_items.keys() - old_items.keys():
            self.events.append({'event': 'item_appeared', 'item_id': item_id, 'name': new_items[item_id]['name']})
        for item_id in old_items.keys() - new_items.keys():
            self.events.append({'event': 'item_disappeared', 'item_id': item_id, 'name': old_items[item_id]['name'],
                                'meaning': 'No completion or delivery inferred from disappearance alone'})
        for item_id in old_items.keys() & new_items.keys():
            first, last = old_items[item_id], new_items[item_id]
            before, after = first.get('cooking'), last.get('cooking')
            if before and after:
                self.rate('cooking', item_id, before.get('elapsed'), after.get('elapsed'), old, new,
                          last.get('layer'), gap or paused)
                if before['state'] != after['state']:
                    self.events.append({'event': 'cooking_state_changed', 'item_id': item_id,
                                        'before': before['state'], 'after': after['state']})
        for first in old['kitchen']['stations']:
            station_id = first['station_id']
            last = new_stations.get(station_id)
            if last is None:
                continue
            before, after = first.get('washing'), last.get('washing')
            if before and after:
                self.rate('washing', station_id, before.get('elapsed'), after.get('elapsed'), old, new,
                          last.get('layer'), gap or paused or before['interactor_count'] != 1 or after['interactor_count'] != 1)
                if (before.get('plate_count') is not None and after.get('plate_count') is not None and
                        after['plate_count'] < before['plate_count']):
                    self.events.append({'event': 'sink_plate_count_decreased', 'station_id': station_id,
                        'before': before['plate_count'], 'after': after['plate_count'],
                        'configured_wash_seconds': after['required']})
            old_item = old_items.get(first.get('item_id'))
            if not old_item or not old_item.get('chopping'):
                continue
            key = (station_id, old_item['item_id'])
            episode = self.chops.get(key)
            if episode is None:
                continue
            a, b = clock(old, first.get('layer')), clock(new, last.get('layer'))
            workers_a = (first.get('work') or {}).get('interactor_count', 0)
            workers_b = (last.get('work') or {}).get('interactor_count', 0)
            if a is not None and b is not None and b >= a:
                delta = b - a
                episode['upper'] += delta if workers_a or workers_b else 0
                episode['lower'] += delta if workers_a and workers_b and not gap else 0
                episode['multiple_workers'] |= max(workers_a, workers_b) > 1
                episode['sampling_gap'] |= gap and bool(workers_a or workers_b)
                episode['interrupted'] |= bool(workers_a and not workers_b and last.get('item_id') == old_item['item_id'])
            new_item = new_items.get(last.get('item_id'))
            if last.get('item_id') != old_item['item_id']:
                expected = old_item['chopping'].get('next_prefab')
                if new_item and expected and new_item['name'].replace('(Clone)', '').strip() == expected:
                    self.events.append({'event': 'chop_replacement_observed', 'station_id': station_id,
                        'raw_item_id': old_item['item_id'], 'prepared_item_id': new_item['item_id'],
                        'prepared_name': new_item['name'], 'active_seconds_bounds': [episode['lower'], episode['upper']],
                        **{k: v for k, v in episode.items() if k not in ('lower', 'upper')}})
                self.chops.pop(key, None)
        self.begin_chops(new)

    def rate(self, kind, identity, before, after, old, new, layer, skip):
        a, b = clock(old, layer), clock(new, layer)
        if not skip and None not in (a, b, before, after) and b > a and after > before:
            self.rates.append({'kind': kind, 'id': identity, 'native_units_per_layer_second': (after - before) / (b - a)})

    def summary(self):
        return {'schema_version': 1, 'samples': self.samples, 'target_interval_seconds': self.interval,
            'sampling_interval_seconds': quantiles(self.intervals), 'request_latency_seconds': quantiles(self.latencies),
            'unity_capture_ms': quantiles(self.capture_ms),
            'gaps_above_3_intervals': sum(v > max(.3, 3 * self.interval) for v in self.intervals),
            'scenes': self.scenes, 'provenance': self.provenance, 'warnings': dict(self.warnings),
            'timing_rates': {kind: quantiles([r['native_units_per_layer_second'] for r in self.rates if r['kind'] == kind])
                             for kind in ('cooking', 'washing', 'plate_return')},
            'configured_plate_return_seconds': self.return_configs,
            'events': self.events, 'last_station_catalogs': self.stations,
            'limits': ['Collider bounds do not verify complete walkability.',
                       'Configured durations and observed durations are distinct.',
                       'Missing samples and item disappearance alone do not establish successful preparation or serving.',
                       'Chop duration bounds assume no unobserved stop/start between adjacent samples; inspect interruptions and gaps.']}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def record(output, seconds=600, interval=.1, port=8765):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    raw = output / 'samples.jsonl'
    if raw.exists():
        raise ValueError('Recording already exists; choose a new output directory')
    evidence = Evidence(interval)
    metadata = {'started_utc': utc(), 'read_only': True, 'target_interval_seconds': interval,
                'max_seconds': seconds, 'recorder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_json(output / 'metadata.json', metadata)
    begun, next_sample = time.monotonic(), time.monotonic()
    last_status, geometries, reason, error = 0., set(), 'duration_limit', None
    try:
        with Client(port) as client, raw.open('x') as samples, (output / 'events.jsonl').open('x') as events:
            while time.monotonic() - begun < seconds:
                if (output / 'STOP').exists():
                    reason = 'stop_file'
                    break
                time.sleep(max(0., next_sample - time.monotonic()))
                sent_utc, sent = utc(), time.monotonic()
                response = client.request('state')
                received = time.monotonic()
                state = response['state']
                if not isinstance(state.get('layer_game_seconds'), list):
                    raise RuntimeError('Install/restart bridge v0.3.0 before recording; timing clocks are missing')
                sample = {'sequence': evidence.samples, 'sent_utc': sent_utc, 'received_utc': utc(),
                          'sent_monotonic': sent, 'received_monotonic': received, 'response': response}
                samples.write(json.dumps(sample, allow_nan=False, separators=(',', ':')) + '\n')
                samples.flush()
                for event in evidence.add(sample):
                    events.write(json.dumps(event, allow_nan=False) + '\n')
                events.flush()
                # Geometry is expensive and redundant; request it once for each loaded kitchen.
                token = state['scene_token']
                if state.get('kitchen', {}).get('available') and token not in geometries:
                    write_json(output / ('geometry-' + token + '.json'), client.request('geometry'))
                    geometries.add(token)
                if received - last_status >= 2:
                    write_json(output / 'status.json', {'samples': evidence.samples, 'scene': state['scene'],
                               'paused': state['paused'], 'elapsed_seconds': received - begun, 'updated_utc': utc()})
                    last_status = received
                next_sample = max(next_sample + interval, time.monotonic())
    except KeyboardInterrupt:
        reason = 'keyboard_interrupt'
    except (OSError, RuntimeError, ValueError) as exc:
        reason, error = 'error', str(exc)
    finally:
        metadata.update(finished_utc=utc(), stop_reason=reason, error=error)
        write_json(output / 'metadata.json', metadata)
        write_json(output / 'summary.json', evidence.summary())
    if error:
        raise RuntimeError(error)
    return evidence.summary()


def analyze(output):
    output = Path(output)
    metadata = json.loads((output / 'metadata.json').read_text())
    evidence = Evidence(metadata['target_interval_seconds'])
    with (output / 'samples.jsonl').open() as stream:
        for line in stream:
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                # A crash can leave the final line incomplete. Never skip corruption in the middle.
                if stream.read():
                    raise
                evidence.warnings['Incomplete final sample discarded'] += 1
                break
            evidence.add(sample)
    write_json(output / 'summary.json', evidence.summary())
    return evidence.summary()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('record')
    p.add_argument('--output', type=Path, default=ROOT / 'outputs' / ('calibration-' + dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')))
    p.add_argument('--seconds', type=float, default=600)
    p.add_argument('--interval', type=float, default=.1)
    p.add_argument('--port', type=int, default=8765)
    p = commands.add_parser('analyze')
    p.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.command == 'record':
        if not math.isfinite(args.interval) or not .1 <= args.interval <= 5:
            parser.error('--interval must be between 0.1 and 5 seconds')
        if not math.isfinite(args.seconds) or not 0 < args.seconds <= 1800:
            parser.error('--seconds must be greater than zero and at most 1800')
        print('Read-only recording:', args.output, flush=True)
        result = record(args.output, args.seconds, args.interval, args.port)
    else:
        result = analyze(args.output)
    print(json.dumps({k: result[k] for k in ('samples', 'sampling_interval_seconds', 'request_latency_seconds', 'unity_capture_ms', 'timing_rates')}, indent=2))


if __name__ == '__main__':
    main()
