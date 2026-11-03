"""Rendered root/subpath pages retain application links, assets, and replay reads."""
from html.parser import HTMLParser
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from app.auth import COOKIE, new_session
from app.config import Settings
from app.main import create_app
from app.models import Base, User


class Paths(HTMLParser):
    def __init__(self):
        super().__init__();self.paths=[]
    def handle_starttag(self, tag, attrs):
        self.paths += [value for key,value in attrs if key in {'href','src','action','data-edit-url','data-delete-url'} and value and value.startswith('/')]


@pytest.mark.parametrize('prefix',['','/sims'])
def test_rendered_pages_and_replay_assets_work_below_configured_prefix(tmp_path,prefix):
    app=create_app(Settings(data_dir=tmp_path,base_url='http://testserver'+prefix,allowed_github_ids=frozenset({123}),reserve_bytes=0))
    Base.metadata.create_all(app.state.engine)
    with app.state.db() as db:
        user=User(github_id=123,login='teammate',display_name='Teammate');db.add(user);db.flush()
        token,session=new_session(db,app.state.settings,user.id)
    source=json.loads((Path(__file__).parent/'fixtures/evaluation-v1.json').read_text(encoding='utf-8'))
    headers={'x-csrf-token':session.csrf_token,'content-type':'application/json'}
    with TestClient(app) as client:
        client.cookies.set(COOKIE,token,path=prefix+'/')
        preview=client.post(prefix+'/api/uploads',content=json.dumps(source),headers=headers)
        assert preview.status_code==200,preview.text
        imported=client.post(prefix+'/api/imports',json={'preview_id':preview.json()['id']},headers=headers)
        assert imported.status_code==200,imported.text
        result=imported.json();assert result['url']==prefix+'/evaluations/'+result['id']
        episode=client.get(prefix+f"/api/evaluations/{result['id']}/episodes/0")
        assert episode.status_code==200,episode.text
        data=episode.json();assert len(data['recording']['steps'][0]['agents'])==3
        run_id=data['participants'][0]['run_id']
        pages=[prefix+'/',prefix+'/?tab=training-runs&q=synthetic',result['url']+'?episode=1&step=2',prefix+'/training-runs/'+run_id,prefix+'/upload']
        for url in pages:
            response=client.get(url);assert response.status_code==200,response.text
            assert f'name="base-path" content="{prefix}"' in response.text
            parser=Paths();parser.feed(response.text)
            assert parser.paths
            for value in parser.paths:
                assert value.startswith(prefix+'/'),(url,value)
                if prefix:assert not value.startswith(prefix+prefix+'/')
            for asset in [value for value in parser.paths if '/static/' in value]:
                static=client.get(asset);assert static.status_code==200,(asset,static.text)
        fragment=client.get(prefix+'/static/replay/viewer.html')
        assert fragment.status_code==200
        assert 'data-replay-asset="kitchen-atlas.png"' in fragment.text
        assert 'src="/static/' not in fragment.text
        for name in ('kitchen-atlas.png','sushi-atlas.png'):
            response=client.get(prefix+'/static/replay/'+name)
            assert response.status_code==200
            assert response.headers['content-type']=='image/png'
    app.state.engine.dispose()
