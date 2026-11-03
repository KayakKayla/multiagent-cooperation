"""Preview/apply the reviewed rowan.gaertner.ca Apache /sims integration.

Run with Python 3.12 as root; default is a read-only diff. --apply backs up files,
enables only required modules, validates, gracefully reloads, and checks HTTPS.
This deliberately supports only the existing, reviewed single-vhost files.
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
import uuid

APACHE = Path("/etc/apache2")
INCLUDE = APACHE / "evals-site.conf"
TARGETS = {
    "tls": ("sites-enabled/rowan.gaertner-le-ssl.conf", "sites-available/rowan.gaertner-le-ssl.conf"),
    "http": ("sites-enabled/rowan.gaertner.conf", "sites-available/rowan.gaertner.conf"),
    "logs": ("conf-enabled/other-vhosts-access-log.conf", "conf-available/other-vhosts-access-log.conf"),
}
MARKER = 'SetEnvIf Request_URI "^/sims(?:/|$)" sims_request'
SAFE_LOG = 'CustomLog ${APACHE_LOG_DIR}/sims-access.log "%h %t \\"%m %U %H\\" %>s %b %L" env=sims_request'


def replace_line(text, old, new):
    pattern = re.compile(r"(?m)^(\s*)" + re.escape(old) + r"[ \t]*$")
    if len(pattern.findall(text)) != 1:
        raise ValueError(f"expected exactly one directive: {old}")
    return pattern.sub(lambda match: match[1] + new, text)


def patch(kind, text):
    if "sims" in text.lower():
        raise ValueError(f"{kind}: existing sims configuration; inspect before reuse")
    if re.search(r"(?im)^\s*(?:GlobalLog|TransferLog|IncludeOptional)\b", text):
        raise ValueError(f"{kind}: unexpected logging/include directive")
    logs = re.findall(r"(?im)^\s*CustomLog\b.*$", text)
    if len(logs) != 1:
        raise ValueError(f"{kind}: expected one CustomLog")
    if kind == "logs":
        if re.search(r"(?im)^\s*<(?:VirtualHost|IfModule)\b", text):
            raise ValueError("expected an unconditional global other-vhosts log")
        actual = [line.strip() for line in text.splitlines()
                  if line.strip() and not line.lstrip().startswith("#")]
        if actual != ["CustomLog ${APACHE_LOG_DIR}/other_vhosts_access.log vhost_combined"]:
            raise ValueError("global log configuration differs from the reviewed original")
        return replace_line(text,
            "CustomLog ${APACHE_LOG_DIR}/other_vhosts_access.log vhost_combined",
            MARKER + "\nCustomLog ${APACHE_LOG_DIR}/other_vhosts_access.log vhost_combined env=!sims_request")
    common = [
        "ServerAdmin webmaster@localhost", "ServerName rowan.gaertner.ca",
        "ServerAlias www.rowan.gaertner.ca", "DocumentRoot /var/www/rowan.gaertner.ca",
        "ErrorLog ${APACHE_LOG_DIR}/error.log", "CustomLog ${APACHE_LOG_DIR}/access.log combined",
    ]
    expected = (["<IfModule mod_ssl.c>", "<VirtualHost *:443>"] + common + [
        "<Directory /var/www/rowan.gaertner.ca>", "Options -Indexes", "</Directory>",
        "SSLCertificateFile /etc/letsencrypt/live/rowan.gaertner.ca/fullchain.pem",
        "SSLCertificateKeyFile /etc/letsencrypt/live/rowan.gaertner.ca/privkey.pem",
        "Include /etc/letsencrypt/options-ssl-apache.conf", "</VirtualHost>", "</IfModule>",
    ]) if kind == "tls" else (["<VirtualHost *:80>"] + common + [
        "RewriteEngine on", "RewriteCond %{SERVER_NAME} =www.rowan.gaertner.ca [OR]",
        "RewriteCond %{SERVER_NAME} =rowan.gaertner.ca",
        "RewriteRule ^ https://%{SERVER_NAME}%{REQUEST_URI} [END,NE,R=permanent]",
        "</VirtualHost>",
    ])
    actual = [line.strip() for line in text.splitlines()
              if line.strip() and not line.lstrip().startswith("#")]
    if actual != expected:
        raise ValueError(f"{kind}: configuration differs from the reviewed original")
    port = "443" if kind == "tls" else "80"
    if re.findall(r"(?im)^\s*<VirtualHost\s+([^>]+)>\s*$", text) != [f"*:{port}"]:
        raise ValueError(f"{kind}: unexpected virtual hosts")
    if len(re.findall(r"(?im)^\s*</VirtualHost>\s*$", text)) != 1:
        raise ValueError(f"{kind}: unexpected closing virtual hosts")
    for directive in ("ServerName rowan.gaertner.ca", "ServerAlias www.rowan.gaertner.ca",
                      "DocumentRoot /var/www/rowan.gaertner.ca"):
        replace_line(text, directive, directive)
    if re.search(r"(?im)^\s*(?:ProxyPass|ProxyPassReverse|SetEnvIf|ErrorLogFormat)\b", text):
        raise ValueError(f"{kind}: unexpected existing proxy/log configuration")
    text = replace_line(text, "CustomLog ${APACHE_LOG_DIR}/access.log combined",
                        "CustomLog ${APACHE_LOG_DIR}/access.log combined env=!sims_request")
    if kind == "tls":
        for directive in (
            "SSLCertificateFile /etc/letsencrypt/live/rowan.gaertner.ca/fullchain.pem",
            "SSLCertificateKeyFile /etc/letsencrypt/live/rowan.gaertner.ca/privkey.pem",
            "Include /etc/letsencrypt/options-ssl-apache.conf",
        ):
            replace_line(text, directive, directive)
        addition = "Include /etc/apache2/evals-site.conf"
    else:
        # HTTP remains an HTTPS redirect, but its original combined log must
        # also exclude callback query strings before that redirect happens.
        addition = MARKER + "\n" + SAFE_LOG
    return replace_line(text, "</VirtualHost>", addition + "\n</VirtualHost>")


def run(*args):
    result = subprocess.run(args, check=False, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    print(result.stdout, end="", flush=True)
    result.check_returncode()
    return result.stdout


def install(path, data, mode=0o644, uid=0, gid=0):
    fd, temporary = tempfile.mkstemp(prefix=".evals-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(data)
            os.fchmod(output.fileno(), mode)
            os.fchown(output.fileno(), uid, gid)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def request(path, *, tls=True):
    scheme, port = ("https", 443) if tls else ("http", 80)
    result = subprocess.run([
        "curl", "--noproxy", "*", "--silent", "--show-error", "--max-time", "15",
        "--resolve", f"rowan.gaertner.ca:{port}:127.0.0.1", "--dump-header", "-",
        f"{scheme}://rowan.gaertner.ca{path}",
    ], check=True, stdout=subprocess.PIPE)
    headers, body = result.stdout.split(b"\r\n\r\n", 1)
    return headers.decode("iso-8859-1"), body


def file_contains(path, token, chunk_size=64 * 1024):
    """Find a byte token using bounded memory, including across read boundaries."""
    if not token:
        raise ValueError("token must not be empty")
    overlap = b""
    with path.open("rb") as source:
        while block := source.read(chunk_size):
            combined = overlap + block
            if token in combined:
                return True
            overlap = combined[-(len(token) - 1):] if len(token) > 1 else b""
    return False


def check_callback_routes(token):
    """Use six fresh HTTPS connections to expose mixed Apache worker routing."""
    queries = [f"code={token}&state={token}",
               f"code={token}&iss=https%3A%2F%2Fgithub.com%2Flogin%2Foauth&state={token}"]
    for _ in range(3):
        for query in queries:
            headers, body = request("/sims/auth/callback?" + query)
            if (" 400 " not in headers.splitlines()[0]
                or "x-content-type-options: nosniff" not in headers.lower()
                or b"Login session expired or missing; sign in again" not in body):
                raise RuntimeError("callback routing check failed; inspect Apache worker generations")
    # HTTP must also avoid logging callback queries before its HTTPS redirect.
    for query in queries:
        request("/sims/auth/callback?" + query, tls=False)


def smoke(root_before):
    headers, body = request("/")
    if " 200 " not in headers.splitlines()[0] or body != root_before:
        raise RuntimeError("existing HTTPS root content changed")
    for attempt in range(20):
        headers, body = request("/sims/")
        if " 200 " in headers.splitlines()[0] and b"/sims/static/site.css" in body:
            break
        if attempt == 19:
            raise RuntimeError("public home/prefix check failed")
        time.sleep(0.25)
    headers, body = request("/sims/static/site.css")
    if " 200 " not in headers.splitlines()[0] or "content-type: text/css" not in headers.lower():
        raise RuntimeError("public stylesheet check failed")
    headers, body = request("/sims/api/evaluations")
    if " 200 " not in headers.splitlines()[0]:
        raise RuntimeError("public API check failed")
    json.loads(body)
    headers, _ = request("/sims")
    if " 301 " not in headers.splitlines()[0] or "location: https://rowan.gaertner.ca/sims/" not in headers.lower():
        raise RuntimeError("public slash redirect check failed")
    token = "evals_privacy_probe_" + uuid.uuid4().hex
    check_callback_routes(token)
    time.sleep(1)
    logs = list(Path("/var/log/apache2").glob("*.log"))
    if not logs:
        raise RuntimeError("no Apache logs found for privacy check")
    for log in logs:
        if file_contains(log, token.encode()):
            raise RuntimeError(f"dummy callback credential leaked to {log}")
    print("PASS HTTPS root unchanged; /sims home, CSS, API, redirect; six callback routes; dummy query absent from Apache logs")


def apply_changes(changes, originals, root_body, inventory):
    enabled_dir = APACHE / "mods-enabled"
    modules_before = set(enabled_dir.iterdir())
    backup = Path(tempfile.mkdtemp(prefix="evals-apache-backup-", dir="/var/backups"))
    for path, (data, metadata) in originals.items():
        install(backup / path.name, data, metadata.st_mode & 0o777, metadata.st_uid, metadata.st_gid)
    print(f"Original configuration backup: {backup}", flush=True)
    try:
        for path, data in changes.items():
            if path in originals:
                metadata = originals[path][1]
                install(path, data, metadata.st_mode & 0o777, metadata.st_uid, metadata.st_gid)
            else:
                install(path, data)
        run("a2enmod", "proxy", "proxy_http", "headers", "setenvif", "alias")
        modules = run("apache2ctl", "-M")
        for name in ("proxy", "proxy_http", "headers", "setenvif", "alias", "log_config"):
            if f"{name}_module" not in modules:
                raise RuntimeError(f"required Apache module missing: {name}")
        run("apache2ctl", "configtest")
        # -S also prints runtime mutexes, which legitimately change when proxy
        # modules load. Compare only its virtual-host section.
        if run("apache2ctl", "-S").split("ServerRoot:", 1)[0] != inventory.split("ServerRoot:", 1)[0]:
            raise RuntimeError("virtual-host inventory changed; inspect configuration")
        run("systemctl", "reload", "apache2")
        run("systemctl", "is-active", "apache2")
        smoke(root_body)
    except BaseException:
        print("Validation failed; restoring original files and module links.", flush=True)
        for path, (data, metadata) in originals.items():
            install(path, data, metadata.st_mode & 0o777, metadata.st_uid, metadata.st_gid)
        INCLUDE.unlink(missing_ok=True)
        for link in set(enabled_dir.iterdir()) - modules_before:
            if not link.is_symlink() or link.resolve().parent != APACHE / "mods-available":
                raise RuntimeError(f"unexpected new module entry; inspect manually: {link}")
            link.unlink()
        run("apache2ctl", "configtest")
        run("systemctl", "reload", "apache2")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    originals = {}
    changes = {}
    for kind, (enabled, available) in TARGETS.items():
        link = APACHE / enabled
        target = APACHE / available
        if not link.is_symlink() or link.resolve(strict=True) != target:
            raise ValueError(f"unexpected enabled-file target: {link}")
        if target.is_symlink() or not target.is_file() or target.stat().st_uid != 0:
            raise ValueError(f"expected root-owned regular file: {target}")
        originals[target] = (target.read_bytes(), target.stat())
        changes[target] = patch(kind, originals[target][0].decode()).encode()
    if INCLUDE.exists() or INCLUDE.is_symlink():
        raise ValueError(f"destination exists: {INCLUDE}")
    changes[INCLUDE] = Path(__file__).with_name("apache-evals-site.conf.example").read_bytes()
    for path, data in changes.items():
        before = originals[path][0].decode() if path in originals else ""
        print("".join(difflib.unified_diff(before.splitlines(True), data.decode().splitlines(True),
                                        fromfile=str(path), tofile=str(path))), end="")
    run("apache2ctl", "configtest")
    inventory = run("apache2ctl", "-S")
    if not args.apply:
        print("Preview only. Review the diff and virtual hosts before using --apply.")
        return
    if os.geteuid() != 0:
        raise PermissionError("--apply requires root")
    root_headers, root_body = request("/")
    if " 200 " not in root_headers.splitlines()[0]:
        raise RuntimeError("existing HTTPS root must return 200 before applying")
    apply_changes(changes, originals, root_body, inventory)


if __name__ == "__main__":
    main()
