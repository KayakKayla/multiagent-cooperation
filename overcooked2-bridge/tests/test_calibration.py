import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1]))
import calibration


def sample(sequence, elapsed=0., cooking=0., paused=False, token='scene', item=True, work=0, prepared=False):
    pot = {'item_id': 2, 'name': 'Pot', 'layer': 0,
           'cooking': {'elapsed': cooking, 'required': 12., 'state': 'Cooked' if cooking >= 12 else 'Raw'}}
    ingredient = {'item_id': 4 if prepared else 3, 'name': 'CutCucumber(Clone)' if prepared else 'Cucumber', 'layer': 0,
                  'chopping': None if prepared else {'completion_fraction': 0., 'next_prefab': 'CutCucumber'}}
    state = {'scene_token': token, 'scene': 's_sushi_1_3', 'paused': paused, 'in_round': True,
             'layer_game_seconds': [elapsed] * 32, 'capture_ms': 2.,
             'kitchen': {'available': True, 'items': [pot] + ([ingredient] if item else []), 'warnings': [],
                         'stations': [{'station_id': 1, 'layer': 0, 'item_id': ingredient['item_id'] if item else None,
                                       'work': {'interactor_count': work}, 'washing': None}]}}
    return {'sequence': sequence, 'sent_monotonic': sequence * .1, 'received_monotonic': sequence * .1 + .01,
            'received_utc': 'test', 'response': {'ok': True, 'plugin_version': '0.3.0', 'assembly_sha256': 'hash', 'state': state}}


class CalibrationTests(unittest.TestCase):
    def test_cooking_rate_uses_game_clock_and_skips_pause_reset_and_scene_change(self):
        e = calibration.Evidence()
        for s in [sample(0), sample(1, .05, .1), sample(2, .05, .1, paused=True),
                  sample(3, .1, .2), sample(4, .2, 0), sample(5, 0, 0, token='next')]:
            e.add(s)
        self.assertEqual(len(e.rates), 1)
        self.assertAlmostEqual(e.rates[0]['native_units_per_layer_second'], 2.)

    def test_chop_replacement_bounds_exclude_idle_and_paused_time(self):
        e = calibration.Evidence()
        for s in [sample(0), sample(1, .1, work=1), sample(2, .2, work=1),
                  sample(3, .2, paused=True, work=1), sample(4, .3, prepared=True)]:
            e.add(s)
        event = next(x for x in e.events if x['event'] == 'chop_replacement_observed')
        self.assertEqual(event['raw_item_id'], 3)
        self.assertEqual(event['prepared_item_id'], 4)
        self.assertFalse(event['left_censored'])
        self.assertAlmostEqual(event['active_seconds_bounds'][0], .1)
        self.assertAlmostEqual(event['active_seconds_bounds'][1], .3)

    def test_item_removal_is_not_assumed_to_be_chop_completion(self):
        e = calibration.Evidence()
        e.add(sample(0, work=1))
        e.add(sample(1, .1, item=False))
        self.assertFalse(any(x['event'] == 'chop_replacement_observed' for x in e.events))

    def test_work_already_started_is_left_censored(self):
        e = calibration.Evidence()
        e.add(sample(0, work=1))
        e.add(sample(1, .1, prepared=True))
        self.assertTrue(next(x for x in e.events if x['event'] == 'chop_replacement_observed')['left_censored'])

    def test_long_sampling_gap_excluded_from_rate_measurements(self):
        e = calibration.Evidence()
        e.add(sample(0))
        last = sample(1, 1., 1.)
        last['sent_monotonic'] = 1.
        last['received_monotonic'] = 1.01
        e.add(last)
        self.assertEqual(e.rates, [])
        self.assertEqual(e.summary()['gaps_above_3_intervals'], 1)

    def test_washing_rate_excludes_multiple_workers(self):
        e = calibration.Evidence()
        for seq in range(3):
            s = sample(seq, seq * .1)
            s['response']['state']['kitchen']['stations'][0]['washing'] = {
                'elapsed': seq * .1, 'required': 3., 'plate_count': 1, 'interactor_count': 1 if seq < 2 else 2}
            e.add(s)
        self.assertEqual(len(e.rates), 1)
        self.assertEqual(e.rates[0]['kind'], 'washing')

    def test_return_countdown_rate_is_positive_and_queue_changes_are_explicit(self):
        e = calibration.Evidence()
        for seq, pending in enumerate([[], [{'station_id': 8, 'remaining_seconds': 5.}],
                                      [{'station_id': 8, 'remaining_seconds': 4.9}], []]):
            s = sample(seq, seq * .1)
            s['response']['state']['kitchen']['plate_return'] = {'configured_seconds': 5., 'pending': pending}
            e.add(s)
        self.assertAlmostEqual(next(r for r in e.rates if r['kind'] == 'plate_return')['native_units_per_layer_second'], 1.)
        self.assertEqual(len([x for x in e.events if x['event'] == 'plate_return_queue_changed']), 2)

    def test_recording_sends_only_read_requests_and_geometry_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            calls = []
            class FakeClient:
                def __init__(self, port):
                    self.sequence = 0
                def __enter__(self):
                    return self
                def __exit__(self, *args):
                    pass
                def request(self, op):
                    calls.append(op)
                    if op == 'geometry':
                        return {'geometry': {'coverage': 'active_collider_world_bounds'}}
                    s = sample(self.sequence, self.sequence * .1)['response']
                    self.sequence += 1
                    if self.sequence == 2:
                        (output / 'STOP').touch()
                    return s
            with patch.object(calibration, 'Client', FakeClient):
                result = calibration.record(output, seconds=1)
            self.assertEqual(calls, ['state', 'geometry', 'state'])
            self.assertEqual(result['samples'], 2)
            self.assertEqual(json.loads((output / 'metadata.json').read_text())['stop_reason'], 'stop_file')
            self.assertEqual(calibration.analyze(output)['samples'], 2)

    def test_disconnect_keeps_completed_samples_and_does_not_reconnect(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(calibration, 'Client') as client:
                client.return_value.__enter__.return_value.request.side_effect = [sample(0)['response'],
                    {'geometry': {}}, OSError('disconnected')]
                with self.assertRaisesRegex(RuntimeError, 'disconnected'):
                    calibration.record(tmp, seconds=1)
                self.assertEqual(client.call_count, 1)
            self.assertEqual(json.loads((Path(tmp) / 'summary.json').read_text())['samples'], 1)
            self.assertEqual(json.loads((Path(tmp) / 'metadata.json').read_text())['stop_reason'], 'error')

    def test_offline_analysis_discards_only_incomplete_final_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / 'metadata.json').write_text('{"target_interval_seconds":0.1}')
            (output / 'samples.jsonl').write_text(json.dumps(sample(0)) + '\n{"partial":')
            self.assertEqual(calibration.analyze(output)['samples'], 1)
            (output / 'samples.jsonl').write_text('{bad\n' + json.dumps(sample(0)) + '\n')
            with self.assertRaises(json.JSONDecodeError):
                calibration.analyze(output)


if __name__ == '__main__':
    unittest.main()
