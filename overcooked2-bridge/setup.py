#!/usr/bin/env python3
"""Project-local dependencies and reversible installation of the OC2 test plugin."""
import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEPS = ROOT / '.deps'
GAME = Path.home() / 'Library/Application Support/Steam/steamapps/common/Overcooked! 2'
BEP_VERSION = '5.4.23.6'
SDK_VERSION = '8.0.426'
BEP_URL = ('https://github.com/BepInEx/BepInEx/releases/download/v' + BEP_VERSION
           + '/BepInEx_macos_universal_' + BEP_VERSION + '.zip')
BEP_SHA = '2e711e21f6570da964fce4654b1976500dbe043f61778bce0008500078564638'
SDK_URL = ('https://builds.dotnet.microsoft.com/dotnet/Sdk/' + SDK_VERSION
           + '/dotnet-sdk-' + SDK_VERSION + '-osx-arm64.tar.gz')
SDK_SHA = ('17cd408c92eb6ceedc06d41fa726c783b9be020c6ae5f3624ad3a0b2c26a78aa7'
           'd09f38338003163a72bb14d437731debf533c0451a5c6a05de3b1e37a3aa50a')
PLUGIN = Path('BepInEx/plugins/Sims.Overcooked2Bridge.dll')
MANIFEST = ROOT / 'outputs/install.json'


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def download(url, path, expected, algorithm='sha256'):
    if not path.exists() or digest(path, algorithm) != expected:
        print('Downloading ' + url, flush=True)
        temporary = path.with_suffix('.download')
        urllib.request.urlretrieve(url, temporary)
        if digest(temporary, algorithm) != expected:
            temporary.unlink()
            raise RuntimeError('Download checksum mismatch: ' + url)
        temporary.replace(path)


def dependencies():
    DEPS.mkdir(exist_ok=True)
    download(BEP_URL, DEPS / 'bepinex.zip', BEP_SHA)
    if not (DEPS / 'bepinex/BepInEx/core/BepInEx.dll').exists():
        with zipfile.ZipFile(DEPS / 'bepinex.zip') as archive:
            archive.extractall(DEPS / 'bepinex')
    download(SDK_URL, DEPS / 'dotnet-sdk.tar.gz', SDK_SHA, 'sha512')
    if not (DEPS / 'dotnet/dotnet').exists():
        destination = DEPS / 'dotnet'
        destination.mkdir(exist_ok=True)
        with tarfile.open(DEPS / 'dotnet-sdk.tar.gz') as archive:
            # These are verified vendor archives. Still reject escaping paths.
            for member in archive.getmembers():
                if member.name.startswith('/') or '..' in Path(member.name).parts:
                    raise RuntimeError('Unsafe archive member')
            archive.extractall(destination)
    print('Dependencies ready (BepInEx ' + BEP_VERSION + ', .NET SDK ' + SDK_VERSION + ').')


def install(game, previous_launch_options):
    if not (game / 'Overcooked2.app').is_dir():
        raise RuntimeError('Game application not found: ' + str(game))
    sources = [(p, p.relative_to(DEPS / 'bepinex'))
               for p in (DEPS / 'bepinex').rglob('*') if p.is_file()]
    sources.append((ROOT / 'bin/Sims.Overcooked2Bridge.dll', PLUGIN))
    prior = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else None
    if prior and prior['game_dir'] != str(game):
        raise RuntimeError('Existing installation manifest refers to another game directory')
    if not sources or any(not src.exists() for src, _ in sources):
        raise RuntimeError('Run setup.py dependencies and bash build.sh first')
    owned = {f['path']: f['sha256'] for f in prior['files']} if prior else {}
    for source, relative in sources:
        target = game / relative
        if target.exists():
            if relative.as_posix() not in owned:
                raise RuntimeError('Refusing to replace pre-existing mod file: ' + str(target))
            if digest(target) != owned[relative.as_posix()]:
                raise RuntimeError('Installed file changed outside this installer: ' + str(target))
    if not prior and (game / 'BepInEx').exists():
        raise RuntimeError('Existing BepInEx installation found; use a clean setup for this test')
    record = prior or {'game_dir': str(game), 'previous_launch_options': previous_launch_options,
                       'bepinex_version': BEP_VERSION, 'files': []}
    # Write ownership after every copy so an interrupted installation is recoverable.
    MANIFEST.parent.mkdir(exist_ok=True)
    for source, relative in sources:
        target = game / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        record['files'] = [f for f in record['files'] if f['path'] != relative.as_posix()]
        record['files'].append({'path': relative.as_posix(), 'sha256': digest(target)})
        MANIFEST.write_text(json.dumps(record, indent=2) + '\n')
    (game / 'run_bepinex.sh').chmod(0o755)
    record['installed_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
    record['plugin_sha256'] = digest(game / PLUGIN)
    MANIFEST.write_text(json.dumps(record, indent=2) + '\n')
    print('Installed. In Steam Properties > General > Launch Options, enter:')
    print('"' + str(game / 'run_bepinex.sh') + '" %command%')
    print('Previous launch option recorded in ' + str(MANIFEST))


def uninstall(game):
    record = json.loads(MANIFEST.read_text())
    if record['game_dir'] != str(game):
        raise RuntimeError('Installation manifest refers to another game directory')
    for entry in record['files']:
        target = game / entry['path']
        if target.exists() and digest(target) != entry['sha256']:
            raise RuntimeError('Refusing to remove modified file: ' + str(target))
    for entry in record['files']:
        (game / entry['path']).unlink(missing_ok=True)
    # Generated BepInEx config/logs are deliberately left available for diagnosis.
    print('Removed installed files. Restore this Steam Launch Options value:')
    print(repr(record['previous_launch_options']))
    MANIFEST.rename(MANIFEST.with_suffix('.removed.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['dependencies', 'build', 'install', 'uninstall', 'check'])
    parser.add_argument('--game-dir', type=Path, default=GAME)
    parser.add_argument('--previous-launch-options', default='')
    args = parser.parse_args()
    try:
        if args.command == 'dependencies':
            dependencies()
        elif args.command == 'build':
            import os
            env = dict(os.environ, OC2_GAME_DIR=str(args.game_dir))
            subprocess.run(['bash', str(ROOT / 'build.sh')], env=env, check=True)
        elif args.command == 'install':
            install(args.game_dir.resolve(), args.previous_launch_options)
        elif args.command == 'uninstall':
            uninstall(args.game_dir.resolve())
        else:
            log = args.game_dir / 'BepInEx/LogOutput.log'
            matches = [line for line in log.read_text(errors='replace').splitlines()
                       if 'SIMS_BRIDGE_' in line]
            print('\n'.join(matches) if matches else 'No bridge startup message found.')
            return 0 if any('SIMS_BRIDGE_READY' in line for line in matches) and not any('SIMS_BRIDGE_FAILED' in line for line in matches) else 1
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, str(exc) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
