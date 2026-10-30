"""Source distribution for the Lightning V3 publication, added 2026-10-30.

SPDX-License-Identifier: AGPL-3.0-only
"""
import hashlib
import html
import io
from pathlib import Path, PurePosixPath
import zipfile


def source_archive(root):
    """Archive only explicitly reviewed files, never the live config or weights."""
    root = Path(root).resolve()
    names = (root / 'SOURCE-FILES.txt').read_text(encoding='utf-8').splitlines()
    if len(names) != len(set(names)):
        raise ValueError('Duplicate source manifest entry')
    required = {'LICENSE', 'NOTICE', 'MODIFICATIONS.md', 'SOURCE-FILES.txt',
                'Dockerfile.lightning', 'README.md'}
    if not required.issubset(names):
        raise ValueError('Incomplete source manifest')
    for path in (root / 'kindi').rglob('*'):
        if path.suffix in ('.py', '.cu', '.c') and path.relative_to(root).as_posix() not in names:
            raise ValueError('Runtime source missing from manifest')
    out = io.BytesIO()
    total = 0
    with zipfile.ZipFile(out, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            rel = PurePosixPath(name)
            if not name or rel.is_absolute() or '..' in rel.parts or '\\' in name or ':' in name:
                raise ValueError('Unsafe source manifest path')
            if any(part in ('.git', '.local', '.env', 'pack', 'state', 'data', 'config') for part in rel.parts):
                raise ValueError('Private directory in source manifest')
            if rel.suffix.lower() in ('.gguf', '.safetensors', '.pth', '.pt', '.key', '.pem', '.conf'):
                raise ValueError('Private config, key or model in source manifest')
            path = root / name
            if any(parent.is_symlink() for parent in [path] + list(path.parents) if parent != root.parent):
                raise ValueError('Symlink in source manifest')
            resolved = path.resolve()
            resolved.relative_to(root)
            size = resolved.stat().st_size
            total += size
            if size > 50 * 1024**2 or total > 100 * 1024**2:
                raise ValueError('Unexpectedly large source distribution')
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 30, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o100755 if name.endswith('.sh') else 0o100644) << 16
            archive.writestr(info, resolved.read_bytes())
    return out.getvalue()


class SourceOfferMiddleware:
    """Public source/legal routes and discoverable links on all HTTP responses.

    Snapshot the shipped source once at startup; rebuild/restart after code changes.
    Inference authentication and streaming bodies pass through unchanged.
    """
    def __init__(self, app, root):
        self.app = app
        self.archive = source_archive(root)
        self.sha256 = hashlib.sha256(self.archive).hexdigest()
        self.license = (Path(root) / 'LICENSE').read_bytes()
        self.notice = (Path(root) / 'NOTICE').read_text(encoding='utf-8')

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http':
            return await self.app(scope, receive, send)
        prefix = scope.get('root_path', '').rstrip('/')
        source = prefix + '/source'
        links = ('<%s/archive>; rel="describedby"; title="Complete corresponding source (AGPL-3.0-only)", '
                 '<%s/license>; rel="license"' % (source, prefix)).encode('ascii')

        async def with_links(message):
            if message['type'] == 'http.response.start':
                message = dict(message)
                message['headers'] = list(message.get('headers', [])) + [(b'link', links)]
            await send(message)

        path = scope.get('path', '/')
        if prefix and path.startswith(prefix + '/'):
            path = path[len(prefix):]
        if scope.get('method') not in ('GET', 'HEAD') or path not in ('/', '/source', '/source/archive', '/license'):
            return await self.app(scope, receive, with_links)
        headers = [(b'cache-control', b'no-store')]
        if path == '/source/archive':
            body = self.archive
            mime = b'application/zip'
            headers += [(b'content-disposition', b'attachment; filename="kindi-lightning-v3-source.zip"')]
        elif path == '/license':
            body = self.license
            mime = b'text/plain; charset=utf-8'
        else:
            body = ('<!doctype html><html lang="en"><meta charset="utf-8">'
                    '<title>Kindi Lightning V3 — source and license</title><main>'
                    '<h1>Kindi 0.3.0 + Lightning backend V3</h1>'
                    '<p>Modified fork maintained by slavalevski-tech, 2026-10-30. '
                    'Original Kindi: Copyright (c) 2026 Amir Husain.</p>'
                    '<p>This software is supplied WITHOUT ANY WARRANTY. You may copy, '
                    'modify and redistribute it under AGPL-3.0-only.</p>'
                    '<p><a href="%s/archive">Download complete corresponding source for this running build</a> '
                    '(free, no API key required). <a href="%s/license">Read the license</a>.</p>'
                    '<p>Source ZIP SHA-256: <code>%s</code></p><h2>Notices</h2><pre>%s</pre></main></html>'
                    % (html.escape(source), html.escape(prefix), self.sha256, html.escape(self.notice))).encode('utf-8')
            mime = b'text/html; charset=utf-8'
        headers += [(b'content-type', mime), (b'content-length', str(len(body)).encode('ascii'))]
        await with_links({'type': 'http.response.start', 'status': 200, 'headers': headers})
        await send({'type': 'http.response.body', 'body': b'' if scope['method'] == 'HEAD' else body})
