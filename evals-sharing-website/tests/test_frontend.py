"""Render real templates and exercise the shared replay packaging boundary."""
from copy import deepcopy
import json
import re
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from jinja2 import Environment, FileSystemLoader, select_autoescape

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "shared"))
from evals_shared.replay import build_web_payload, read_replay, write_replay


def recording():
    return {"schema_version": 1, "metadata": {"layout": "example", "mode": "trained", "seed": 1}, "steps": [
        {"step": 0, "grid": [[[0, 0, 0]] * 3], "recipe": "onion", "pots": [],
         "agents": [{"id": f"agent_{i}", "index": i, "color": color, "facing": "right", "position": [i, 0], "inventory": 0, "holding": "nothing", "action": None} for i, color in enumerate(("red", "blue", "purple"))],
         "totals": {"team_reward": 0, "correct_dishes": 0, "wrong_dishes": 0}}]}


class FrontendTests(unittest.TestCase):
    def setUp(self):
        self.env = Environment(loader=FileSystemLoader(ROOT / "templates"), autoescape=select_autoescape())

    def test_selected_recording_adapts_three_agents_without_frames_or_mutation(self):
        data = recording(); before = deepcopy(data)
        participants = [{"agent_id": f"agent_{i}", "label": f"Policy {i} · {i * 10} steps"} for i in range(3)]
        output = build_web_payload(data, participants)["recording"]
        self.assertTrue(output["presentation"]["supported"])
        self.assertEqual(output["participants"], participants)
        self.assertNotIn("frames", output)
        self.assertEqual(data, before)
        self.assertEqual(build_web_payload(None), {"recording": None})

    def test_unsupported_tiles_report_reason_without_reconstructing_state(self):
        data = recording(); data["steps"][0]["grid"][0][0] = [8, 0, 0]
        output = build_web_payload(data)["recording"]
        self.assertFalse(output["presentation"]["supported"])
        self.assertIn("tile", output["presentation"]["reason"])
        self.assertEqual(output["steps"], data["steps"])

    def test_malformed_or_unknown_renderer_is_unsupported_and_preserves_exact_state(self):
        for renderer in ('invalid', None, [], {'kind':'unsupported-future'}, {'kind':None}):
            with self.subTest(renderer=renderer):
                data=recording(); data['metadata']['renderer']=renderer; before=deepcopy(data)
                output=build_web_payload(data)['recording']
                self.assertFalse(output['presentation']['supported'])
                self.assertIn('renderer',output['presentation']['reason'])
                self.assertEqual(output['presentation']['scenes'],[])
                self.assertEqual(output['steps'],before['steps'])
                self.assertEqual(data,before)

    def test_legacy_absent_renderer_still_infers_supported_overcooked(self):
        data=recording()
        self.assertNotIn('renderer',data['metadata'])
        output=build_web_payload(data)['recording']['presentation']
        self.assertTrue(output['supported'])
        self.assertEqual(output['kind'],'overcooked')

    def test_offline_viewer_uses_shared_code_and_keeps_inert_json(self):
        data = recording(); data["metadata"]["title"] = '</script><script>alert(1)</script>'
        data["steps"][0]["agents"] = data["steps"][0]["agents"][:2]
        data["frames"] = ["data:image/png;base64,AA=="]
        with TemporaryDirectory() as directory:
            target = Path(directory) / 'replay.html'; write_replay(data, target)
            html = target.read_text(encoding='utf-8')
            self.assertNotIn(data["metadata"]["title"], html)
            self.assertIn((ROOT/'static/replay/viewer.js').read_text(encoding='utf-8'), html)
            self.assertNotIn('src="/static/', html)
            self.assertEqual(read_replay(target)["steps"], data["steps"])
            self.assertEqual(read_replay(target)["frames"], data["frames"])

    def test_detail_template_escapes_uploaded_annotations_and_labels_exact_checkpoints(self):
        evil = '<img src=x onerror=alert(1)>'
        evaluation = {"id":"eval-1","title":evil,"notes":evil,"version":1,"episode_count":2,"layout":"demo", "summary":{"mean_team_reward":5},"protocol":{},"environment":{},"provenance":{},"device":{},"participants":[{"agent_id":"agent_0","run_id":"run-1","label":"Baseline · 500 steps","sha256":"a"*64,"observation_settings":{},"action_settings":{}}],"episodes":[{"index":0,"seed":1,"steps":0,"metrics":{"team_reward":0},"recording_available":True},{"index":1,"seed":2,"steps":10,"metrics":{"team_reward":10},"recording_available":False}]}
        html=self.env.get_template('evaluation.html').render(evaluation=evaluation,user={"login":"teammate"},csrf_token='csrf')
        self.assertNotIn(evil,html)
        self.assertIn('&lt;img',html)
        self.assertIn('Baseline · 500 steps',html)
        self.assertIn('Recording unavailable',html)
        self.assertIn('Evaluation-wide statistics',html)
        self.assertIn('Selected episode',html)
        self.assertIn('value="1"',html)
        anonymous=self.env.get_template('evaluation.html').render(evaluation=evaluation,user=None,csrf_token='')
        self.assertNotIn('data-delete-url=',anonymous)
        self.assertNotIn('data-edit-url=',anonymous)

    def test_home_tabs_filters_and_empty_state_render(self):
        html=self.env.get_template('home.html').render(user=None,csrf_token='',tab='training-runs',page=2,pages=3,filters={'q':'a&b','layout':'x','algorithm':'ppo'},evaluations=[],training_runs=[])
        self.assertIn('Training runs',html)
        self.assertIn('No training runs match',html)
        self.assertIn('q=a%26b',html)
        self.assertIn('page=1',html)
        self.assertIn('page=3',html)

    def test_sushi_mean_reward_and_source_evaluation_time_render_on_home(self):
        item={"id":"e1", "title":"Sushi", "layout":"sushi", "episode_count":2,
              "summary":{"mean_raw_team_reward":140}, "evaluated_at":"2026-09-01T00:00:00Z",
              "created_at":"2026-11-01T00:00:00Z"}
        html=self.env.get_template('home.html').render(user=None,csrf_token='',tab='evaluations',page=1,pages=1,
            filters={'q':'','layout':'','algorithm':''},evaluations=[item],training_runs=[])
        self.assertIn('<td>140</td>',html)
        self.assertIn('2026-09-01T00:00:00Z',html)
        self.assertNotIn('2026-11-01T00:00:00Z',html)

    def test_offline_atlas_backgrounds_do_not_depend_on_adjacent_files(self):
        data=recording(); data['frames']=['data:image/png;base64,AA==']
        with TemporaryDirectory() as directory:
            target=Path(directory)/'replay.html';write_replay(data,target)
            html=target.read_text(encoding='utf-8')
            css=html.split('<style>',1)[1].split('</style>',1)[0]
            # Website-only relative atlas backgrounds must not match an offline root.
            for selector,body in re.findall(r'([^{}]+)\{([^{}]*)\}',css):
                if re.search(r'background-image:\s*url\("?(?:kitchen|sushi)-atlas\.png',body):
                    self.assertIn('.embedded',selector)
            self.assertIn('.replay-viewer .portrait{background-image:url("${atlas.src}")}',html)
            self.assertIn('id="sprite-atlas" src="data:image/png;base64,',html)
            self.assertIn('id="sushi-atlas" src="data:image/png;base64,',html)

    def test_client_annotation_and_delete_behaviors(self):
        subprocess.run(['node','--test',str(ROOT/'tests/frontend-behavior.cjs')],check=True,capture_output=True)

    def test_client_upload_checkpoint_suggestions_require_reviewed_run(self):
        subprocess.run(['node','--test',str(ROOT/'tests/upload-behavior.cjs')],check=True,capture_output=True)

    def test_client_selected_episode_prefix_and_artwork_paths(self):
        subprocess.run(['node','--test',str(ROOT/'tests/episode-behavior.cjs')],check=True,capture_output=True)

    def test_javascript_parses(self):
        for script in (ROOT/'static').rglob('*.js'):
            subprocess.run(['node','--check',str(script)],check=True,capture_output=True)


if __name__ == '__main__':
    unittest.main()
