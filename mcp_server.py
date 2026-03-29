"""claude-timecard MCP Server: 3ツール構成.

ユーザー向けツール:
  1. timecard_day     — 特定日の全部入りビュー
  2. timecard_conversation — 会話内容の切り出し
  3. timecard_register — 同義語/ストップワード登録

MCP stdio transportはstdoutをJSON-RPCに使うため、
lib/内のprint()がstdoutに出るとプロトコルが壊れてハングする。
全ツール実行時にstdoutをstderrにリダイレクトして防止する。
"""

import contextlib
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# lib をインポートパスに追加
sys.path.insert(0, str(Path(__file__).parent))

from lib.config import TimecardConfig, JST, ATTRIBUTION_SHORT
from lib.parser.events import Event, collect_events, parse_timestamp
from lib.analysis.blocks import Block, build_blocks, subdivide_blocks_by_keywords, format_duration
from lib.analysis.tfidf import (
    build_tfidf, extract_keywords, _tokenize,
    _add_stopwords, add_synonym, import_sudachi_synonyms,
)
from lib.analysis.tasks import infer_tasks_branch_first
from lib.timing import Timer

mcp = FastMCP("claude-timecard")

# --- constants ---

_PROJECTS_DIR = Path.home() / ".claude" / "projects"
_META = {"tool": ATTRIBUTION_SHORT, "license": "AGPL-3.0"}


# --- internal helpers ---

def _parse_date(s: str) -> datetime:
    if s.endswith("d"):
        days = int(s[:-1])
        return datetime.now(JST).replace(
            hour=0, minute=0, second=0, microsecond=0
        ) - timedelta(days=days)
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=JST)


def _load_data(
    start: str = "7d",
    end: str | None = None,
    project: str | None = None,
    config: TimecardConfig | None = None,
) -> tuple[list[Event], list[Block], dict[str, float]]:
    """共通のデータロード処理."""
    if config is None:
        config = TimecardConfig()

    date_start = _parse_date(start)
    date_end = _parse_date(end) + timedelta(days=1) if end else datetime.now(JST) + timedelta(days=1)

    events = collect_events(_PROJECTS_DIR, date_start, date_end, project)
    if not events:
        return [], [], {}

    blocks = build_blocks(events, config.idle_threshold_min, config.per_turn_time_min)
    tfidf = build_tfidf(blocks)

    # ノイズ除去
    n_blocks = len(blocks)
    if n_blocks >= 4:
        block_vocabs = []
        for b in blocks:
            vocab = set()
            for msg in b.messages:
                vocab.update(_tokenize(msg))
            block_vocabs.append(vocab)
        df: Counter = Counter()
        for v in block_vocabs:
            for w in v:
                df[w] += 1
        noise_words = {w for w, c in df.items() if c / n_blocks > 0.3}
        tfidf = {w: s for w, s in tfidf.items() if w not in noise_words}

    blocks = subdivide_blocks_by_keywords(blocks, tfidf)
    return events, blocks, tfidf


def _block_branch_summary(branches: list[str]) -> str:
    meaningful = [
        b for b in branches if b and b not in ("main", "master", "HEAD", "development")
    ]
    if not meaningful:
        return ""
    counts = Counter(meaningful)
    top = counts.most_common(1)
    return top[0][0] if top else ""


def _resolve_project_dir(project_name: str) -> Path | None:
    """プロジェクト名からgitリポジトリのパスを推定."""
    # Claude Codeのプロジェクトdir名: C--Users-tatut-Documents-xxx → C:\Users\tatut\Documents\xxx
    for proj_dir in _PROJECTS_DIR.iterdir():
        if not proj_dir.is_dir():
            continue
        if project_name.lower() in proj_dir.name.lower():
            # dir名からパスを復元
            name = proj_dir.name
            # C--Users-xxx → C:\Users\xxx
            path_str = re.sub(r"^([A-Z])--", r"\1:\\", name)
            path_str = path_str.replace("-", "\\")
            # ハイフンを含むディレクトリ名のために、存在チェックしながら復元
            candidate = Path(path_str)
            if candidate.exists():
                return candidate
            # fallback: よくあるパターンを試す
            # Documents配下を探す
            docs = Path.home() / "Documents"
            for d in docs.iterdir():
                if d.is_dir() and project_name.lower() in d.name.lower():
                    return d
    return None


def _get_git_commits(repo_dir: Path, branch: str, date: str) -> list[dict]:
    """指定日・ブランチのgitコミットを取得."""
    try:
        result = subprocess.run(
            ["git", "log", f"--after={date} 00:00", f"--before={date} 23:59",
             "--all", f"--grep=", "--format=%H|%s|%an|%ai",
             "--", "."],
            capture_output=True, text=True, timeout=10,
            cwd=str(repo_dir),
        )
        if result.returncode != 0:
            return []

        # ブランチ名でフィルタ（完全一致は難しいのでコミットメッセージ+ブランチ含む）
        # 代わりにその日の全コミットを取得してブランチ名をgrepで絞る
        result2 = subprocess.run(
            ["git", "log", f"--after={date} 00:00", f"--before={date} 23:59",
             "--all", "--format=%H|%s|%an|%ai", "--numstat"],
            capture_output=True, text=True, timeout=10,
            cwd=str(repo_dir),
        )
        if result2.returncode != 0:
            return []

        commits = []
        current_commit = None
        for line in result2.stdout.split("\n"):
            if "|" in line and len(line.split("|")) == 4:
                parts = line.split("|")
                current_commit = {
                    "hash": parts[0][:8],
                    "message": parts[1],
                    "author": parts[2],
                    "date": parts[3].strip(),
                    "files": [],
                }
                commits.append(current_commit)
            elif current_commit and line.strip() and "\t" in line:
                parts = line.split("\t")
                if len(parts) >= 3:
                    current_commit["files"].append({
                        "added": parts[0],
                        "deleted": parts[1],
                        "path": parts[2],
                    })

        # ブランチ名に関連するコミットをフィルタ（ベストエフォート）
        branch_short = branch.split("/")[-1] if "/" in branch else branch
        filtered = []
        for c in commits:
            # ブランチ名の一部がコミットメッセージに含まれる or 全部返す
            if branch_short.lower() in c["message"].lower() or not branch_short:
                filtered.append(c)

        return filtered[:10] if filtered else commits[:5]
    except Exception:
        return []


def _get_pr_info(repo_dir: Path, pr_numbers: list[int]) -> list[dict]:
    """GitHub PRの情報（タイトル、コメント）を取得."""
    results = []
    for pr_num in pr_numbers[:3]:  # 最大3件
        try:
            result = subprocess.run(
                ["gh", "pr", "view", str(pr_num), "--json",
                 "title,body,comments,reviews,url"],
                capture_output=True, text=True, timeout=15,
                cwd=str(repo_dir),
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                pr_info = {
                    "number": pr_num,
                    "title": data.get("title", ""),
                    "url": data.get("url", ""),
                    "body": (data.get("body", "") or "")[:500],
                    "comments": [
                        {"author": c.get("author", {}).get("login", ""),
                         "body": c.get("body", "")[:300]}
                        for c in (data.get("comments", []) or [])[:5]
                    ],
                    "reviews": [
                        {"author": r.get("author", {}).get("login", ""),
                         "state": r.get("state", ""),
                         "body": (r.get("body", "") or "")[:300]}
                        for r in (data.get("reviews", []) or [])[:5]
                    ],
                }
                results.append(pr_info)
        except Exception:
            pass
    return results


def _generate_excerpt(
    blocks: list[Block],
    block_idxs: list[int],
    branch: str,
    keywords: list[str],
    commits: list[dict],
    pr_info: list[dict],
) -> str:
    """Haikuで作業要約を生成."""
    # メッセージ収集
    messages = []
    total_chars = 0
    for bi in block_idxs:
        b = blocks[bi]
        for msg in b.messages:
            if total_chars + len(msg) > 30000:
                break
            messages.append(f"[{b.start.strftime('%H:%M')}] {msg}")
            total_chars += len(msg)

    messages_text = "\n".join(messages)

    # コミット情報
    commit_text = ""
    if commits:
        commit_lines = []
        for c in commits[:5]:
            files_str = ", ".join(f["path"] for f in c.get("files", [])[:5])
            commit_lines.append(f"- {c['hash']} {c['message']} ({files_str})")
        commit_text = "\n## コミット\n" + "\n".join(commit_lines)

    # PR情報
    pr_text = ""
    if pr_info:
        pr_lines = []
        for pr in pr_info:
            pr_lines.append(f"- PR #{pr['number']}: {pr['title']}")
            if pr.get("body"):
                pr_lines.append(f"  説明: {pr['body'][:200]}")
            for c in pr.get("comments", []):
                pr_lines.append(f"  @{c['author']}: {c['body'][:150]}")
            for r in pr.get("reviews", []):
                pr_lines.append(f"  Review({r['state']}) @{r['author']}: {r['body'][:150]}")
        pr_text = "\n## PR\n" + "\n".join(pr_lines)

    time_range = f"{blocks[block_idxs[0]].start.strftime('%H:%M')}~{blocks[block_idxs[-1]].end.strftime('%H:%M')}"
    dur = sum((blocks[bi].end - blocks[bi].start).total_seconds() / 60 for bi in block_idxs)

    prompt = (
        "以下の作業セッションを2-3文で簡潔に要約してください。\n"
        "具体的に何をしたか、結果はどうだったかを含めてください。\n\n"
        f"ブランチ: {branch}\n"
        f"時間: {time_range} ({format_duration(dur)})\n"
        f"キーワード: {', '.join(keywords)}\n"
        f"{commit_text}\n{pr_text}\n\n"
        f"## ユーザーメッセージ\n{messages_text}\n\n"
        "要約のみを出力してください。"
    )

    try:
        result = subprocess.run(
            ["claude", "-p", "--model", "haiku"],
            input=prompt,
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:500]
    except Exception:
        pass
    return ""


def _collect_raw_messages(
    date: str,
    start_time: str | None,
    end_time: str | None,
    project: str | None,
) -> list[dict]:
    """JSONLから指定期間の生メッセージを収集."""
    date_start = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=JST)
    if start_time:
        h, m = start_time.split(":")
        date_start = date_start.replace(hour=int(h), minute=int(m))

    date_end = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=JST) + timedelta(days=1)
    if end_time:
        h, m = end_time.split(":")
        date_end = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=JST).replace(
            hour=int(h), minute=int(m)
        )

    messages = []
    for proj_dir in sorted(_PROJECTS_DIR.iterdir()):
        if not proj_dir.is_dir():
            continue
        proj_name = proj_dir.name
        short = re.sub(r"C--Users-[^-]+-Documents-", "", proj_name)
        short = re.sub(r"C--Users-[^-]+-", "~/", short)

        if project and project.lower() not in short.lower():
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
                            if not isinstance(content, str) or len(content.strip()) < 4:
                                continue

                            branch = d.get("gitBranch", "") or ""
                            messages.append({
                                "time": ts.strftime("%H:%M"),
                                "branch": branch.split("/")[-1] if "/" in branch else branch,
                                "text": content[:500],
                            })
                        except (json.JSONDecodeError, KeyError):
                            pass
            except OSError:
                continue

    messages.sort(key=lambda x: x["time"])
    return messages


# =====================================================================
# MCP Tools (3 user-facing tools)
# =====================================================================


@mcp.tool()
def timecard_day(
    date: str,
    project: str | None = None,
    generate_excerpts: bool = True,
) -> str:
    """特定の日の作業時間を、ブランチ別にグループ化して返す。

    各ブランチグループにはキーワード、PR情報（コメント含む）、コミット情報（ファイル変更含む）、
    AIによる作業要約が含まれる。タイムシート作成に最適。

    Args:
        date: 対象日 (YYYY-MM-DD形式)。例: "2026-03-27"
        project: プロジェクト名フィルタ (部分一致)。例: "prefab"
        generate_excerpts: trueならAI要約を生成（Haiku使用、ブランチ数×1回）。falseならスキップ
    """
    with contextlib.redirect_stdout(sys.stderr):
        events, blocks, tfidf = _load_data(date, date, project)
        if not events:
            return json.dumps({"_meta": _META, "error": "該当日のデータがありません"}, ensure_ascii=False)

        # ブランチ別にブロックをグループ化
        branch_groups: dict[str, list[int]] = defaultdict(list)
        for i, b in enumerate(blocks):
            branch = _block_branch_summary(b.branches) or "main"
            branch_groups[branch].append(i)

        # プロジェクトディレクトリ推定（git/PR情報用）
        project_dirs: list[Path] = []
        if project:
            repo = _resolve_project_dir(project)
            if repo:
                project_dirs.append(repo)

        # 各ブランチグループの情報を構築
        groups = []
        excerpt_jobs = []

        for branch, idxs in sorted(branch_groups.items(), key=lambda x: -len(x[1])):
            group_blocks = [blocks[i] for i in idxs]
            active_min = sum((b.end - b.start).total_seconds() / 60 for b in group_blocks)
            time_ranges = []
            # 連続ブロックをまとめて時間帯にする
            current_start = group_blocks[0].start
            current_end = group_blocks[0].end
            for b in group_blocks[1:]:
                gap = (b.start - current_end).total_seconds() / 60
                if gap > 30:
                    time_ranges.append(f"{current_start.strftime('%H:%M')}~{current_end.strftime('%H:%M')}")
                    current_start = b.start
                current_end = b.end
            time_ranges.append(f"{current_start.strftime('%H:%M')}~{current_end.strftime('%H:%M')}")

            # キーワード
            all_kw: Counter = Counter()
            for b in group_blocks:
                for w, _ in extract_keywords(b.messages, 5, tfidf):
                    all_kw[w] += 1
            top_keywords = [w for w, _ in all_kw.most_common(8)]

            # プロジェクト
            all_projects = sorted({p for b in group_blocks for p in b.projects})

            # PR番号
            all_prs = sorted({n for b in group_blocks for n in b.pr_numbers})

            # コミット/PR情報（ベストエフォート）
            commits = []
            pr_info = []
            for repo in project_dirs:
                commits = _get_git_commits(repo, branch, date)
                if all_prs:
                    pr_info = _get_pr_info(repo, all_prs)
                break

            group = {
                "branch": branch,
                "projects": all_projects,
                "active_hours": round(active_min / 60, 1),
                "active_minutes": round(active_min, 1),
                "time_ranges": time_ranges,
                "blocks": len(idxs),
                "turns": sum(b.turns for b in group_blocks),
                "keywords": top_keywords,
                "pr_numbers": all_prs,
                "pr_info": pr_info,
                "commits": commits[:10],
                "ai_excerpt": None,
            }
            groups.append(group)

            if generate_excerpts:
                excerpt_jobs.append((branch, idxs, top_keywords, commits, pr_info, len(groups) - 1))

        # AI excerpt生成（並列）
        if excerpt_jobs:
            def _run_excerpt(job):
                branch, idxs, kws, commits, prs, idx = job
                excerpt = _generate_excerpt(blocks, idxs, branch, kws, commits, prs)
                return idx, excerpt

            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(_run_excerpt, job) for job in excerpt_jobs]
                for future in as_completed(futures):
                    try:
                        idx, excerpt = future.result()
                        groups[idx]["ai_excerpt"] = excerpt
                    except Exception:
                        pass

        # 全体サマリー
        total_active = sum(g["active_minutes"] for g in groups)

    return json.dumps({
        "_meta": _META,
        "date": date,
        "active_hours": round(total_active / 60, 1),
        "active_minutes": round(total_active, 1),
        "total_blocks": sum(g["blocks"] for g in groups),
        "total_turns": sum(g["turns"] for g in groups),
        "branch_groups": groups,
    }, ensure_ascii=False, indent=2)


@mcp.tool()
def timecard_conversation(
    date: str,
    project: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
) -> str:
    """指定日の会話内容（ユーザーメッセージ）を時系列で返す。

    タイムシートの作業内容を正確に記述するための根拠データ。
    AI要約では不十分な場合に、実際の会話内容を確認するために使う。

    Args:
        date: 対象日 (YYYY-MM-DD)
        project: プロジェクト名フィルタ
        start_time: 開始時刻 (HH:MM)。省略時はその日の最初から
        end_time: 終了時刻 (HH:MM)。省略時はその日の最後まで
    """
    with contextlib.redirect_stdout(sys.stderr):
        messages = _collect_raw_messages(date, start_time, end_time, project)

    return json.dumps({
        "_meta": _META,
        "date": date,
        "time_range": f"{start_time or '00:00'}~{end_time or '23:59'}",
        "message_count": len(messages),
        "messages": messages[:200],  # 最大200件
    }, ensure_ascii=False, indent=2)


@mcp.tool()
def timecard_register(
    type: str,
    words: list[str] | None = None,
    canonical: str | None = None,
    aliases: list[str] | None = None,
    sudachi_path: str | None = None,
) -> str:
    """キーワード精度を改善するための登録ツール。

    同義語、ストップワード、Sudachi辞書インポートの3つの機能を提供。

    使い方:
    - ストップワード追加: type="stopword", words=["ultrathink", "一旦"]
    - 同義語追加: type="synonym", canonical="deploy", aliases=["デプロイ"]
    - Sudachi辞書インポート: type="sudachi", sudachi_path="https://..."

    Args:
        type: "stopword" / "synonym" / "sudachi"
        words: ストップワードのリスト (type="stopword" 時)
        canonical: 同義語の正規形 (type="synonym" 時)
        aliases: 同義語の別名リスト (type="synonym" 時)
        sudachi_path: Sudachi synonyms.txt のパスまたはURL (type="sudachi" 時)
    """
    with contextlib.redirect_stdout(sys.stderr):
        if type == "stopword":
            if not words:
                return json.dumps({"error": "words を指定してください"}, ensure_ascii=False)
            _add_stopwords(words)
            return json.dumps({
                "_meta": _META,
                "action": "stopword_added",
                "words": words,
                "message": f"{len(words)}語をストップワードに追加しました",
            }, ensure_ascii=False, indent=2)

        elif type == "synonym":
            if not canonical or not aliases:
                return json.dumps({"error": "canonical と aliases を指定してください"}, ensure_ascii=False)
            add_synonym(canonical, aliases)
            return json.dumps({
                "_meta": _META,
                "action": "synonym_added",
                "canonical": canonical,
                "aliases": aliases,
                "message": f"同義語登録: {canonical} ← {', '.join(aliases)}",
                "saved_to": "~/.config/claude-timecard/synonyms.json",
            }, ensure_ascii=False, indent=2)

        elif type == "sudachi":
            if not sudachi_path:
                return json.dumps({"error": "sudachi_path を指定してください"}, ensure_ascii=False)
            count = import_sudachi_synonyms(sudachi_path)
            return json.dumps({
                "_meta": _META,
                "action": "sudachi_imported",
                "groups_imported": count,
                "message": f"Sudachi同義語辞書から{count}グループをインポートしました",
                "saved_to": "~/.config/claude-timecard/synonyms.json",
            }, ensure_ascii=False, indent=2)

        else:
            return json.dumps({"error": f"不明なtype: {type}。stopword/synonym/sudachi のいずれかを指定"}, ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="stdio")
