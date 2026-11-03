"""Bounded Apache migration guards and failure recovery; no host changes."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import subprocess

import pytest


spec = importlib.util.spec_from_file_location(
    "install_apache", Path(__file__).resolve().parents[1] / "deploy/install_apache.py")
apache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(apache)

COMMON = """ServerAdmin webmaster@localhost
ServerName rowan.gaertner.ca
ServerAlias www.rowan.gaertner.ca
DocumentRoot /var/www/rowan.gaertner.ca
ErrorLog ${APACHE_LOG_DIR}/error.log
CustomLog ${APACHE_LOG_DIR}/access.log combined
"""
TLS = "<IfModule mod_ssl.c>\n<VirtualHost *:443>\n" + COMMON + """
<Directory /var/www/rowan.gaertner.ca>
Options -Indexes
</Directory>
SSLCertificateFile /etc/letsencrypt/live/rowan.gaertner.ca/fullchain.pem
SSLCertificateKeyFile /etc/letsencrypt/live/rowan.gaertner.ca/privkey.pem
Include /etc/letsencrypt/options-ssl-apache.conf
</VirtualHost>
</IfModule>
"""
HTTP = "<VirtualHost *:80>\n" + COMMON + """RewriteEngine on
RewriteCond %{SERVER_NAME} =www.rowan.gaertner.ca [OR]
RewriteCond %{SERVER_NAME} =rowan.gaertner.ca
RewriteRule ^ https://%{SERVER_NAME}%{REQUEST_URI} [END,NE,R=permanent]
</VirtualHost>
"""
LOG = "# inherited logging\nCustomLog ${APACHE_LOG_DIR}/other_vhosts_access.log vhost_combined\n"


def test_patch_preserves_existing_site_and_excludes_callback_queries():
    tls = apache.patch("tls", TLS)
    http = apache.patch("http", HTTP)
    logs = apache.patch("logs", LOG)
    assert "Include /etc/apache2/evals-site.conf\n</VirtualHost>" in tls
    assert "SSLCertificateFile /etc/letsencrypt/live/rowan.gaertner.ca/fullchain.pem" in tls
    assert "RewriteRule ^ https://%{SERVER_NAME}%{REQUEST_URI} [END,NE,R=permanent]" in http
    for result in (tls, http, logs):
        assert "env=!sims_request" in result
    for result in (http, logs):
        assert apache.MARKER in result
    assert '%U' in apache.SAFE_LOG and '%r' not in apache.SAFE_LOG
    assert '\\"%m %U %H\\"' in http


@pytest.mark.parametrize("kind,source", [("tls", TLS), ("http", HTTP), ("logs", LOG)])
@pytest.mark.parametrize("extra", ["GlobalLog /tmp/extra combined\n",
                                  "CustomLog /tmp/extra combined\n",
                                  "Include /tmp/extra\n"])
def test_unreviewed_directives_fail_closed(kind, source, extra):
    with pytest.raises(ValueError):
        apache.patch(kind, source + extra)


def test_existing_log_conditions_and_repeat_application_are_rejected():
    with pytest.raises(ValueError):
        apache.patch("tls", TLS.replace("access.log combined", "access.log combined env=existing"))
    with pytest.raises(ValueError):
        apache.patch("tls", apache.patch("tls", TLS))


def test_failed_command_prints_diagnostic_before_raising(monkeypatch, capsys):
    monkeypatch.setattr(apache.subprocess, "run", lambda *args, **kwargs:
                        subprocess.CompletedProcess(args[0], 1, "Syntax error: unknown directive\n"))
    with pytest.raises(subprocess.CalledProcessError):
        apache.run("apache2ctl", "configtest")
    assert "Syntax error: unknown directive" in capsys.readouterr().out


def test_log_scan_detects_token_across_chunk_boundary(tmp_path):
    log = tmp_path / "access.log"
    log.write_bytes(b"1234567secret-token-rest")
    assert apache.file_contains(log, b"secret-token", chunk_size=8)
    assert not apache.file_contains(log, b"missing-token", chunk_size=8)
    assert apache.file_contains(log, b"1", chunk_size=8)


@pytest.mark.parametrize("failure", ["configtest", "reload", "smoke"])
def test_failure_restores_files_and_validates_before_reloading(tmp_path, monkeypatch, failure):
    (tmp_path / "mods-enabled").mkdir()
    preexisting = tmp_path / "mods-enabled" / "ssl.load"
    preexisting.write_bytes(b"preexisting module")
    newly_enabled = tmp_path / "mods-enabled" / "proxy.load"
    # Model a2enmod's symlink without requiring Windows symlink privileges.
    original_is_symlink = Path.is_symlink
    original_resolve = Path.resolve
    monkeypatch.setattr(Path, "is_symlink", lambda path:
                        True if path == newly_enabled else original_is_symlink(path))
    monkeypatch.setattr(Path, "resolve", lambda path, *args, **kwargs:
                        tmp_path / "mods-available" / "proxy.load" if path == newly_enabled
                        else original_resolve(path, *args, **kwargs))
    backup = tmp_path / "backup"
    backup.mkdir()
    original = tmp_path / "original.conf"
    original.write_bytes(b"original")
    include = tmp_path / "evals-site.conf"
    monkeypatch.setattr(apache, "APACHE", tmp_path)
    monkeypatch.setattr(apache, "INCLUDE", include)
    monkeypatch.setattr(apache.tempfile, "mkdtemp", lambda **kw: str(backup))
    monkeypatch.setattr(apache, "install", lambda path, data, *args: path.write_bytes(data))
    commands = []
    failed = False

    def run(*args):
        nonlocal failed
        commands.append(args)
        if args[0] == "a2enmod":
            newly_enabled.write_bytes(b"new module link")
        target = args == (("apache2ctl", "configtest") if failure == "configtest"
                          else ("systemctl", "reload", "apache2"))
        if failure != "smoke" and target and not failed:
            failed = True
            raise RuntimeError("injected failure")
        if args == ("apache2ctl", "-M"):
            return " ".join(f"{name}_module" for name in
                            ("proxy", "proxy_http", "headers", "setenvif", "alias", "log_config"))
        return "inventory"

    monkeypatch.setattr(apache, "run", run)
    monkeypatch.setattr(apache, "smoke", lambda body: (_ for _ in ()).throw(RuntimeError("smoke failure")))
    metadata = SimpleNamespace(st_mode=0o640, st_uid=0, st_gid=0)
    with pytest.raises(RuntimeError):
        apache.apply_changes({original: b"changed", include: b"include"},
                             {original: (b"original", metadata)}, b"home", "inventory")
    assert original.read_bytes() == b"original"
    assert (backup / original.name).read_bytes() == b"original"
    assert not include.exists()
    assert preexisting.read_bytes() == b"preexisting module"
    assert not newly_enabled.exists()
    assert commands[-2:] == [("apache2ctl", "configtest"), ("systemctl", "reload", "apache2")]
