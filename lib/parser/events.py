"""Event dataclass, collect_events, parse_timestamp."""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..config import JST


@dataclass
class Event:
    """1つのユーザーメッセージイベント."""

    ts: datetime
    project: str
    content: str
    branch: str = ""
    pr_numbers: list[int] = field(default_factory=list)


def parse_timestamp(ts_str: str) -> datetime | None:
    """ISO timestamp文字列をJST datetimeに変換."""
    try:
        ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return ts.astimezone(JST)
    except (ValueError, TypeError):
        return None


def _clean_content(content: str) -> str:
    """メタテキストを除去."""
    content = re.sub(
        r"<system-reminder>.*?</system-reminder>",
        "",
        content,
        flags=re.DOTALL,
    )
    content = re.sub(
        r"<task-notification>.*?</task-notification>",
        "",
        content,
        flags=re.DOTALL,
    )
    content = re.sub(
        r"<teammate-message[^>]*>.*?</teammate-message>",
        "",
        content,
        flags=re.DOTALL,
    )
    content = re.sub(
        r"\[関連する過去の記憶\].*",
        "",
        content,
        flags=re.DOTALL,
    )
    content = re.sub(
        r"\[前回セッションの要約\].*",
        "",
        content,
        flags=re.DOTALL,
    )
    content = re.sub(
        r"C:\\Users\\[^\s]+",
        "",
        content,
    )
    # /home/user/... 形式のパス
    content = re.sub(
        r"/(?:home|Users)/\S+",
        "",
        content,
    )
    # その他メタテキスト除去
    content = re.sub(
        r"<[a-z_-]+(?:\s[^>]*)?>.*?</[a-z_-]+>",
        "",
        content,
        flags=re.DOTALL,
    )
    return content.strip()


def collect_events(
    projects_dir: Path,
    date_start: datetime,
    date_end: datetime,
    project_filter: str | None = None,
) -> list[Event]:
    """全プロジェクトからユーザーメッセージイベントを収集."""
    events: list[Event] = []

    for proj_dir in sorted(projects_dir.iterdir()):
        if not proj_dir.is_dir():
            continue
        proj_name = proj_dir.name
        # プロジェクトディレクトリ名からユーザーパスを除去して短縮名にする
        import re as _re
        short = _re.sub(r"C--Users-[^-]+-Documents-", "", proj_name)
        short = _re.sub(r"C--Users-[^-]+-", "~/", short)

        if project_filter and project_filter.lower() not in short.lower():
            continue

        for jsonl_path in proj_dir.glob("*.jsonl"):
            if jsonl_path.stat().st_size < 500:
                continue

            try:
                with open(jsonl_path, encoding="utf-8", errors="replace") as f:
                    for line in f:
                        try:
                            d = json.loads(line)
                            if d.get("type") != "user":
                                continue
                            ts = parse_timestamp(d.get("timestamp", ""))
                            if ts is None or ts < date_start or ts >= date_end:
                                continue

                            msg = d.get("message", {})
                            content = msg.get("content", "")
                            if isinstance(content, list):
                                texts = [
                                    c.get("text", "")
                                    for c in content
                                    if isinstance(c, dict) and c.get("type") == "text"
                                ]
                                content = " ".join(texts)
                            if not (
                                isinstance(content, str) and len(content.strip()) > 3
                            ):
                                continue

                            content = _clean_content(content)
                            if len(content) < 4:
                                continue

                            branch = d.get("gitBranch", "") or ""
                            pr_numbers = [
                                int(n) for n in re.findall(r"#(\d{2,4})", content)
                            ]

                            events.append(
                                Event(
                                    ts=ts,
                                    project=short,
                                    content=content[:300].replace("\n", " "),
                                    branch=branch,
                                    pr_numbers=pr_numbers,
                                )
                            )
                        except (json.JSONDecodeError, KeyError):
                            pass
            except OSError:
                continue

    events.sort(key=lambda x: x.ts)
    return events
