"""JSONL読み込みユーティリティ.

主要なJSONL解析ロジックは events.py の collect_events に統合。
ここでは将来的な拡張用のヘルパーを提供。
"""

import json
from pathlib import Path


def iter_jsonl(path: Path):
    """JSONLファイルを1行ずつイテレート."""
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
