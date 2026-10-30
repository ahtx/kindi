"""CPU-only source offer, isolation and ASGI pass-through tests, 2026-10-30.
SPDX-License-Identifier: AGPL-3.0-only
"""
import asyncio
import io
from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile
from kindi.source_offer import SourceOfferMiddleware, source_archive

ROOT = Path(__file__).resolve().parent.parent


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for name in ('LICENSE', 'NOTICE', 'MODIFICATIONS.md', 'README.md', 'Dockerfile.lightning'):
            shutil.copyfile(str(ROOT / name), str(self.root / name))
        self.names = ['LICENSE', 'NOTICE', 'MODIFICATIONS.md', 'README.md', 'Dockerfile.lightning', 'SOURCE-FILES.txt']
        self.write_manifest()
        self.loop = asyncio.new_event_loop()

    def tearDown(self):
        self.loop.close()
        self.temp.cleanup()

    def write_manifest(self):
        (self.root / 'SOURCE-FILES.txt').write_text('\n'.join(self.names) + '\n', encoding='utf-8')

    def call(self, middleware, path, method='GET', prefix=''):
        sent = []
        async def send(message):
            sent.append(message)
        async def receive():
            return {'type': 'http.request', 'body': b'', 'more_body': False}
        self.loop.run_until_complete(middleware({'type': 'http', 'path': path,
            'root_path': prefix, 'method': method, 'headers': []}, receive, send))
        return sent

    @staticmethod
    async def downstream(scope, receive, send):
        await send({'type': 'http.response.start', 'status': 401, 'headers': [(b'x-test', b'kept')]})
        await send({'type': 'http.response.body', 'body': b'data: one\n\n', 'more_body': True})
        await send({'type': 'http.response.body', 'body': b'data: [DONE]\n\n', 'more_body': False})

    def test_only_manifest_files_no_live_credentials_or_weights(self):
        (self.root / '.local').mkdir()
        (self.root / '.local/kindi.conf').write_text('private test config')
        (self.root / 'model.gguf').write_bytes(b'test model')
        with zipfile.ZipFile(io.BytesIO(source_archive(self.root))) as archive:
            self.assertEqual(set(archive.namelist()), set(self.names))
            self.assertEqual(archive.read('LICENSE'), (ROOT / 'LICENSE').read_bytes())

    def test_archive_stable_and_matches_shipped_source(self):
        first = source_archive(ROOT)
        self.assertEqual(first, source_archive(ROOT))
        with zipfile.ZipFile(io.BytesIO(first)) as archive:
            for name in archive.namelist():
                self.assertEqual(archive.read(name), (ROOT / name).read_bytes())

    def test_public_routes_do_not_require_inference_auth(self):
        app = SourceOfferMiddleware(self.downstream, self.root)
        for path in ('/', '/source', '/source/archive', '/license'):
            response = self.call(app, path)
            self.assertEqual(response[0]['status'], 200)
            self.assertIn(b'link', dict(response[0]['headers']))
        self.assertIn(b'WITHOUT ANY WARRANTY', self.call(app, '/source')[1]['body'])
        self.assertEqual(self.call(app, '/license')[1]['body'], (ROOT / 'LICENSE').read_bytes())
        self.assertEqual(self.call(app, '/source/archive', 'HEAD')[1]['body'], b'')

    def test_auth_streaming_and_errors_pass_through(self):
        app = SourceOfferMiddleware(self.downstream, self.root)
        response = self.call(app, '/v1/chat/completions', 'POST')
        self.assertEqual(response[0]['status'], 401)
        self.assertEqual(dict(response[0]['headers'])[b'x-test'], b'kept')
        self.assertEqual([m['body'] for m in response[1:]], [b'data: one\n\n', b'data: [DONE]\n\n'])
        self.assertTrue(response[1]['more_body'])

    def test_proxy_prefix(self):
        app = SourceOfferMiddleware(self.downstream, self.root)
        result = self.call(app, '/backend/source', prefix='/backend')
        self.assertIn(b'/backend/source/archive', result[1]['body'])

    def test_unsafe_or_private_manifest_entries_fail_closed(self):
        for name in ('../escape.txt', '/absolute', 'C:/secret', '.local/password.txt', 'model.gguf', 'kindi.conf'):
            self.names.append(name)
            self.write_manifest()
            with self.assertRaises((ValueError, OSError)):
                source_archive(self.root)
            self.names.pop()

    def test_missing_runtime_source_fails_closed(self):
        (self.root / 'kindi').mkdir()
        (self.root / 'kindi/new_backend.py').write_text('# added source')
        with self.assertRaises(ValueError):
            source_archive(self.root)


if __name__ == '__main__':
    unittest.main()
