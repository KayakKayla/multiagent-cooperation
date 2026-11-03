"""Public selected-episode API preserves unsupported recordings as readable evidence."""
import copy
import pytest
from test_backend import doc, site, publish


@pytest.mark.parametrize('renderer', ['invalid', None, [], {'kind':'unsupported-future'}, {'kind':None}])
def test_selected_episode_unknown_renderer_returns_readable_unsupported_recording(site, doc, renderer):
    original=copy.deepcopy(doc['episodes'][0]['recording'])
    original['metadata']['renderer']=renderer
    doc['episodes'][0]['recording']=original
    result=publish(site,doc)
    response=site[1].get(f"/api/evaluations/{result['id']}/episodes/0")
    assert response.status_code==200,response.text
    output=response.json()['recording']
    assert output['steps']==original['steps']
    assert output['metadata']==original['metadata']
    assert output['presentation']['supported'] is False
    assert 'renderer' in output['presentation']['reason']
    assert output['presentation']['scenes']==[]


def test_selected_episode_legacy_missing_renderer_infers_known_state(site, doc):
    recording=doc['episodes'][0]['recording']
    del recording['metadata']['renderer']
    for state in recording['steps']:
        state['grid']=[[[0,0,0] for _ in row] for row in state['grid']]
        for agent,color in zip(state['agents'],('red','blue','purple')):
            agent.update(color=color,inventory=0)
    result=publish(site,doc)
    response=site[1].get(f"/api/evaluations/{result['id']}/episodes/0")
    assert response.status_code==200,response.text
    output=response.json()['recording']
    assert output['steps']==recording['steps']
    assert output['presentation']['supported'] is True
    assert output['presentation']['kind']=='overcooked'
