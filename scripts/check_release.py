"""Check the source-only release before packaging; added 2026-10-30.
SPDX-License-Identifier: AGPL-3.0-only
"""
import ast
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from kindi.source_offer import source_archive


def main():
    data = source_archive(ROOT)
    issues = []
    patterns = {
        'private-key': r'-----BEGIN [A-Z ]*PRIVATE KEY-----',
        'service-token': r'(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|sk-proj-[A-Za-z0-9_-]{25,})',
        'private-host': r'\b(?:10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+)\b',
        'host-identity': r'(?:GPU-[0-9a-fA-F-]{20,}|/home/(?!user/|example/)[^/\s]+/|[A-Z]:[/\\]Users[/\\][^/\\\s]+[/\\])',
    }
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for name in archive.namelist():
            content = archive.read(name)
            if name.endswith('.py'):
                ast.parse(content, filename=name)
            try:
                text = content.decode('utf-8')
            except UnicodeDecodeError:
                continue
            for kind, pattern in patterns.items():
                # The scanner itself includes literal diagnostic examples.
                if name == 'scripts/check_release.py':
                    continue
                for number, line in enumerate(text.splitlines(), 1):
                    if re.search(pattern, line):
                        issues.append({'path': name, 'line': number, 'type': kind})
        count = len(archive.namelist())
    if issues:
        print(json.dumps({'passed': False, 'issues': issues}, indent=2))
        return 1
    print(json.dumps({'passed': True, 'files': count, 'zip_bytes': len(data),
                      'sha256': hashlib.sha256(data).hexdigest(),
                      'scope': 'manifest, source paths, syntax, selected secret and host patterns; not a universal secret detector'}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
