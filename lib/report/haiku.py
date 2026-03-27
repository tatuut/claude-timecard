"""claude_query, cache管理, segment/task/synthesis レポート生成."""

import hashlib
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ..config import (
    C_BOLD, C_CYAN, C_DIM, C_GREEN, C_MAGENTA, C_RESET,
)
from ..analysis.blocks import Block, SubBlock, format_duration
from ..analysis.tasks import TaskNode

_CACHE_DIR = Path.home() / ".cache" / "claude-timecard" / "reports"


def _cache_key(content: str, model: str) -> str:
    """コンテンツ + モデル名からSHA256キャッシュキーを生成."""
    h = hashlib.sha256(f"{model}:{content}".encode("utf-8")).hexdigest()[:16]
    return h


def _cache_get(key: str) -> str | None:
    """キャッシュからレポートを取得。"""
    path = _CACHE_DIR / f"{key}.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def _cache_put(key: str, value: str) -> None:
    """レポートをキャッシュに保存."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / f"{key}.txt"
    path.write_text(value, encoding="utf-8")


def claude_query(prompt: str, model: str = "haiku", timeout: int = 120) -> str:
    """claude -p でプロンプトを実行して結果を返す."""
    try:
        result = subprocess.run(
            ["claude", "-p", "--model", model],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return f"[エラー: returncode={result.returncode}] {result.stderr.strip()}"
    except subprocess.TimeoutExpired:
        return "[タイムアウト]"
    except FileNotFoundError:
        return "[claude CLI が見つかりません]"


def _block_branch_summary(branches: list[str]) -> str:
    """ブロック内の代表ブランチを返す."""
    meaningful = [
        b for b in branches if b and b not in ("main", "master", "HEAD", "development")
    ]
    if not meaningful:
        return ""
    counts = Counter(meaningful)
    top = counts.most_common(2)
    parts = top[0][0].split("/")
    if len(parts) >= 3:
        result = "/".join(parts[2:])
    elif len(parts) == 2:
        result = parts[1]
    else:
        result = top[0][0]
    if len(top) > 1:
        result += f" +{len(top) - 1}"
    return result


def _collect_block_messages(
    block_idxs: list[int],
    blocks: list[Block],
    max_chars: int = 50000,
) -> str:
    """指定ブロックの全メッセージを結合."""
    parts: list[str] = []
    total = 0
    for bi in block_idxs:
        b = blocks[bi]
        header = (
            f"--- {b.start.strftime('%m/%d %H:%M')}〜{b.end.strftime('%H:%M')}"
            f" ({(b.end - b.start).total_seconds() / 60:.0f}分)"
            f" [{', '.join(sorted(b.projects))}]"
            f" {_block_branch_summary(b.branches) or ''} ---"
        )
        parts.append(header)
        total += len(header)
        for msg in b.messages:
            if total + len(msg) > max_chars:
                parts.append("[...残り省略...]")
                return "\n".join(parts)
            parts.append(msg)
            total += len(msg)
    return "\n".join(parts)


def _collect_block_messages_structured(
    block_idxs: list[int],
    blocks: list[Block],
    sub_blocks: dict[int, list[SubBlock]] | None = None,
    max_chars: int = 50000,
) -> str:
    """サブブロック構造付きでメッセージを収集.

    sub_blocksがある場合、各ブロックのサブブロック構造をプロンプトに含める。
    """
    parts: list[str] = []
    total = 0

    for bi in block_idxs:
        b = blocks[bi]
        header = (
            f"--- {b.start.strftime('%m/%d %H:%M')}〜{b.end.strftime('%H:%M')}"
            f" ({(b.end - b.start).total_seconds() / 60:.0f}分)"
            f" [{', '.join(sorted(b.projects))}]"
            f" {_block_branch_summary(b.branches) or ''} ---"
        )
        parts.append(header)
        total += len(header)

        # サブブロック構造がある場合
        if sub_blocks and bi in sub_blocks:
            subs = sub_blocks[bi]
            for pi, sb in enumerate(subs, 1):
                kw_str = ", ".join(sb.keywords[:5]) if sb.keywords else ""
                trans_tag = " [transition zone]" if sb.is_transition else ""
                phase_header = (
                    f"--- Phase {pi}"
                    f" ({sb.start.strftime('%H:%M')}-{sb.end.strftime('%H:%M')},"
                    f" keywords: {kw_str}){trans_tag} ---"
                )
                parts.append(phase_header)
                total += len(phase_header)
                for msg in sb.messages:
                    if total + len(msg) > max_chars:
                        parts.append("[...残り省略...]")
                        return "\n".join(parts)
                    parts.append(msg)
                    total += len(msg)
        else:
            for msg in b.messages:
                if total + len(msg) > max_chars:
                    parts.append("[...残り省略...]")
                    return "\n".join(parts)
                parts.append(msg)
                total += len(msg)

    return "\n".join(parts)


def _segment_task_blocks(
    task: TaskNode,
    blocks: list[Block],
    gap_minutes: int = 120,
) -> list[list[int]]:
    """タスクのブロック集合を時間ギャップでセグメントに分割."""
    sorted_idxs = sorted(task.block_set, key=lambda i: blocks[i].start)
    if not sorted_idxs:
        return []

    segments: list[list[int]] = [[sorted_idxs[0]]]
    for bi in sorted_idxs[1:]:
        prev_end = blocks[segments[-1][-1]].end
        curr_start = blocks[bi].start
        gap = (curr_start - prev_end).total_seconds() / 60
        if gap > gap_minutes:
            segments.append([bi])
        else:
            segments[-1].append(bi)
    return segments


def ai_classify_blocks(blocks: list[Block], model: str = "haiku"):
    """claude -p で各ブロックのタスク内容をAI分類する."""
    print(f"\n{C_BOLD}  AI タスク分類 (model: {model}){C_RESET}")
    print(f"  {'-' * 70}")

    for b in blocks:
        dur = (b.end - b.start).total_seconds() / 60
        if dur < 5:
            continue

        branch_str = _block_branch_summary(b.branches)
        proj_str = ", ".join(sorted(b.projects))

        sample = b.messages[:5]
        sample_text = "\n".join(m[:100] for m in sample)

        prompt = (
            "以下はClaude Codeでの作業セッションの断片です。\n"
            f"プロジェクト: {proj_str}\n"
            f"ブランチ: {branch_str or 'N/A'}\n"
            f"時間: {b.start.strftime('%m/%d %H:%M')}〜{b.end.strftime('%H:%M')} ({dur:.0f}分)\n"
            f"ユーザーメッセージ({len(b.messages)}件中{len(sample)}件):\n"
            f"{sample_text}\n\n"
            "このセッションで行われた作業を15文字以内の日本語で1つ要約してください。"
            "例: 「パーサーバグ修正」「UXオンボーディング実装」「コードレビュー対応」"
            "\n要約のみを出力してください。"
        )

        try:
            result = subprocess.run(
                ["claude", "-p", "--model", model],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=30,
            )
            label = result.stdout.strip()[:30] if result.returncode == 0 else "?"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            label = "?"

        time_str = f"{b.start.strftime('%m/%d %H:%M')}〜{b.end.strftime('%H:%M')}"
        print(
            f"  {C_CYAN}{time_str}{C_RESET}"
            f" ({dur:.0f}m)"
            f"  {C_MAGENTA}{label}{C_RESET}"
            f"  {C_DIM}{proj_str}{C_RESET}"
        )


def _build_segment_job(
    ti: int,
    si: int,
    seg_idxs: list[int],
    blocks: list[Block],
    kw_str: str,
    model: str,
    use_cache: bool,
) -> dict:
    """Phase 1用のセグメントジョブを構築。キャッシュ確認も行う。"""
    seg_start = blocks[seg_idxs[0]].start
    seg_end = blocks[seg_idxs[-1]].end
    seg_dur = sum(
        (blocks[bi].end - blocks[bi].start).total_seconds() / 60
        for bi in seg_idxs
    )
    seg_time = (
        f"{seg_start.strftime('%m/%d %H:%M')}~"
        f"{seg_end.strftime('%m/%d %H:%M')}"
    )
    messages_text = _collect_block_messages(seg_idxs, blocks)
    cache_k = _cache_key(messages_text, f"seg_{model}") if use_cache else None
    cached = _cache_get(cache_k) if cache_k else None

    prompt = (
        "あなたはソフトウェア開発の作業ログアナリストです。\n"
        "以下はClaude Code（AIペアプログラミングツール）での"
        "作業セッションログの一部分です。\n"
        "ユーザーのメッセージを読み、この作業セグメントを簡潔に"
        "レポートしてください。\n\n"
        f"## セグメント情報\n"
        f"親タスク: {kw_str}\n"
        f"期間: {seg_time}\n"
        f"作業時間: {format_duration(seg_dur)}\n"
        f"ブロック数: {len(seg_idxs)}\n\n"
        f"## ユーザーメッセージ全文\n"
        f"{messages_text}\n\n"
        "## 出力フォーマット（日本語・簡潔に）\n"
        "### やったこと\n"
        "（1-2文で具体的に何をしたか）\n\n"
        "### 結果\n"
        "（成果・到達点を箇条書き1-3項目）\n\n"
        "### 課題・次のアクション\n"
        "（残課題や次にやるべきことがあれば1-2項目。なければ省略）\n"
    )

    return {
        "ti": ti,
        "si": si,
        "seg_time": seg_time,
        "seg_dur": seg_dur,
        "seg_idxs": seg_idxs,
        "prompt": prompt,
        "cache_k": cache_k,
        "cached": cached,
    }


def _execute_segment_job(job: dict, model: str) -> dict:
    """1つのセグメントジョブを実行（キャッシュミス時のみAPI呼び出し）."""
    if job["cached"]:
        return {**job, "report": job["cached"], "from_cache": True}

    report_text = claude_query(job["prompt"], model=model, timeout=180)
    if job["cache_k"] and not report_text.startswith("["):
        _cache_put(job["cache_k"], report_text)
    return {**job, "report": report_text, "from_cache": False}


def generate_report(
    tasks: list[TaskNode],
    blocks: list[Block],
    context_keywords: list[str] | None = None,
    model: str = "haiku",
    output: str = "timecard_report.md",
    use_cache: bool = True,
    max_workers: int = 5,
):
    """3階層AIレポートを生成。Phase 1（セグメント）は並列実行。"""
    if not tasks:
        print(f"\n  {C_DIM}レポート対象のタスクがありません{C_RESET}")
        return

    print(f"\n{C_BOLD}{'=' * 70}{C_RESET}")
    print(f"  {C_BOLD}AIレポート生成 (model: {model}, 並列: {max_workers}){C_RESET}")
    print(f"{C_BOLD}{'=' * 70}{C_RESET}")

    ctx_str = ", ".join(context_keywords[:6]) if context_keywords else "なし"
    task_results: list[dict] = []
    cache_hits = 0
    api_calls = 0

    # --- Phase 1: 全タスクの全セグメントジョブを収集 ---
    all_jobs: list[dict] = []
    task_segment_map: dict[int, tuple[str, str, int]] = {}  # ti → (kw_str, dur_str, n_seg)

    for ti, task in enumerate(tasks, 1):
        kw_str = ", ".join(task.keywords)
        dur_str = format_duration(task.active_minutes)
        segments = _segment_task_blocks(task, blocks)
        task_segment_map[ti] = (kw_str, dur_str, len(segments))

        for si, seg_idxs in enumerate(segments, 1):
            job = _build_segment_job(ti, si, seg_idxs, blocks, kw_str, model, use_cache)
            all_jobs.append(job)

    # キャッシュ済み / 未済を分離
    cached_jobs = [j for j in all_jobs if j["cached"]]
    uncached_jobs = [j for j in all_jobs if not j["cached"]]
    cache_hits += len(cached_jobs)

    total_jobs = len(all_jobs)
    print(
        f"\n  Phase 1: {total_jobs}セグメント"
        f" ({len(cached_jobs)} cached, {len(uncached_jobs)} to generate)"
    )

    # キャッシュ済みを先に結果に入れる
    results_by_key: dict[tuple[int, int], dict] = {}
    for j in cached_jobs:
        print(
            f"    {C_DIM}[{j['ti']}-{j['si']}] {j['seg_time']}"
            f" ({len(j['seg_idxs'])}blk, {format_duration(j['seg_dur'])})"
            f" {C_GREEN}cached{C_RESET}"
        )
        results_by_key[(j["ti"], j["si"])] = {**j, "report": j["cached"], "from_cache": True}

    # 未キャッシュを並列実行
    if uncached_jobs:
        print(
            f"    {C_CYAN}Generating {len(uncached_jobs)} segments"
            f" ({max_workers} workers)...{C_RESET}"
        )
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_execute_segment_job, job, model): job
                for job in uncached_jobs
            }
            for future in as_completed(futures):
                result = future.result()
                api_calls += 1
                results_by_key[(result["ti"], result["si"])] = result
                print(
                    f"    {C_DIM}[{result['ti']}-{result['si']}]"
                    f" {result['seg_time']}"
                    f" {C_GREEN}done{C_RESET}"
                )

    # --- タスクごとにseg_reportsを組み立て ---
    for ti, task in enumerate(tasks, 1):
        kw_str, dur_str, n_seg = task_segment_map[ti]
        segments = _segment_task_blocks(task, blocks)

        print(
            f"\n  {C_BOLD}Task {ti}: {kw_str}{C_RESET}"
            f"  {C_DIM}({dur_str}, {len(task.block_set)}blk -> {n_seg}seg){C_RESET}"
        )

        seg_reports: list[dict] = []
        for si in range(1, n_seg + 1):
            r = results_by_key[(ti, si)]
            seg_reports.append(
                {
                    "index": si,
                    "time": r["seg_time"],
                    "duration": format_duration(r["seg_dur"]),
                    "blocks": len(r["seg_idxs"]),
                    "report": r["report"],
                }
            )

        # --- Phase 2: セグメント群からタスクまとめ ---
        seg_summaries = "\n\n---\n\n".join(
            f"### セグメント {s['index']}: {s['time']} ({s['duration']})\n\n"
            f"{s['report']}"
            for s in seg_reports
        )

        task_cache_k = _cache_key(seg_summaries, f"task_{model}") if use_cache else None
        task_cached = _cache_get(task_cache_k) if task_cache_k else None

        if task_cached:
            cache_hits += 1
            print(f"    {C_CYAN}[Task {ti} まとめ]{C_RESET} {C_GREEN}cached{C_RESET}")
            task_summary = task_cached
        else:
            print(
                f"    {C_CYAN}[Task {ti} まとめ]{C_RESET} ...",
                end="",
                flush=True,
            )
            task_prompt = (
                "あなたはソフトウェア開発の作業ログアナリストです。\n"
                "以下は1つのタスクを構成する複数の作業セグメントのレポートです。\n"
                "これらを統合して、このタスク全体のまとめを作成してください。\n\n"
                f"## タスク情報\n"
                f"キーワード: {kw_str}\n"
                f"総作業時間: {dur_str}\n"
                f"セグメント数: {n_seg}\n\n"
                f"## セグメントレポート\n\n{seg_summaries}\n\n"
                "## 出力フォーマット（日本語で）\n"
                "### タスク名\n"
                "（15文字以内の具体的なタスク名）\n\n"
                "### 概要\n"
                "（2-3文でこのタスクの全体像。何が起きて、何が達成されたか）\n\n"
                "### 進捗\n"
                "（完了/進行中/中断 + 根拠1文）\n\n"
                "### タイムライン\n"
                "（各セグメントで何が進んだかを時系列1行ずつ）\n\n"
                "### 主な成果\n"
                "（箇条書き2-5項目）\n\n"
                "### 残課題\n"
                "（あれば箇条書き。なければ「なし」）\n"
            )
            task_summary = claude_query(task_prompt, model=model, timeout=180)
            api_calls += 1
            if task_cache_k and not task_summary.startswith("["):
                _cache_put(task_cache_k, task_summary)
            print(f" {C_GREEN}done{C_RESET}")

        task_results.append(
            {
                "index": ti,
                "keywords": kw_str,
                "duration": dur_str,
                "blocks": len(task.block_set),
                "segments": seg_reports,
                "summary": task_summary,
            }
        )

    # --- Phase 3: 統合レポート ---
    all_task_summaries = "\n\n---\n\n".join(
        f"## Task {r['index']}: {r['keywords']} "
        f"({r['duration']}, {r['blocks']}blk, {len(r['segments'])}seg)\n\n"
        f"{r['summary']}"
        for r in task_results
    )

    union_blocks: set[int] = set()
    for t in tasks:
        union_blocks |= t.block_set
    union_min = sum(
        (blocks[bi].end - blocks[bi].start).total_seconds() / 60 for bi in union_blocks
    )
    total_segs = sum(len(r["segments"]) for r in task_results)

    synth_cache_k = (
        _cache_key(all_task_summaries, f"synth_{model}") if use_cache else None
    )
    synth_cached = _cache_get(synth_cache_k) if synth_cache_k else None

    if synth_cached:
        cache_hits += 1
        print(f"\n  {C_CYAN}[統合]{C_RESET} {C_GREEN}cached{C_RESET}")
        general_report = synth_cached
    else:
        print(
            f"\n  {C_CYAN}[統合]{C_RESET} 全タスクを統合中...",
            end="",
            flush=True,
        )
        synthesis_prompt = (
            "あなたはソフトウェア開発チームのマネージャーです。\n"
            "以下は期間中の各タスクのまとめレポートです。\n"
            "これらを統合して、この期間の作業全体を俯瞰するレポートを作成して"
            "ください。読む人にとって価値のある洞察を含めてください。\n\n"
            f"## 期間統計\n"
            f"総Active時間: {format_duration(union_min)}\n"
            f"タスク数: {len(tasks)}\n"
            f"セグメント数: {total_segs}\n"
            f"文脈キーワード: {ctx_str}\n\n"
            f"## 各タスクのまとめ\n\n{all_task_summaries}\n\n"
            "## 出力フォーマット（日本語・Markdownで）\n"
            "# 作業レポート\n\n"
            "## エグゼクティブサマリー\n"
            "（3-5文。この期間で最も重要だったこと、全体の進捗感）\n\n"
            "## タスク一覧\n"
            "（各タスクを1行で要約。時間・状態を含める）\n\n"
            "## 注目ポイント\n"
            "（タスク間の依存関係、ボトルネック、リスク、"
            "特に効率が良かった/悪かった点など、マネージャーが知りたい情報）\n\n"
            "## 推奨アクション\n"
            "（次にやるべきことの優先順位付きリスト）\n"
        )
        general_report = claude_query(synthesis_prompt, model=model, timeout=180)
        api_calls += 1
        if synth_cache_k and not general_report.startswith("["):
            _cache_put(synth_cache_k, general_report)
        print(f" {C_GREEN}done{C_RESET}")

    # --- Markdownファイル出力（3階層: 統合→タスク→セグメント）---
    out_path = Path(output).resolve()
    out_dir = out_path.parent
    stem = out_path.stem

    # Layer 1: セグメントファイル
    seg_files: list[Path] = []
    for r in task_results:
        ti = r["index"]
        for s in r["segments"]:
            si = s["index"]
            seg_path = out_dir / f"{stem}_task{ti}_seg{si}.md"
            seg_md = (
                f"# Task {ti} - セグメント {si}: {s['time']}\n"
                f"*{s['duration']} / {s['blocks']}ブロック*\n\n"
                f"**親タスク:** {r['keywords']}\n\n"
                f"---\n\n"
                f"{s['report']}\n\n"
                f"---\n"
                f"[<- Task {ti} に戻る]({stem}_task{ti}.md)\n"
            )
            seg_path.write_text(seg_md, encoding="utf-8")
            seg_files.append(seg_path)

    # Layer 2: タスクファイル
    task_files: list[Path] = []
    for r in task_results:
        ti = r["index"]
        task_path = out_dir / f"{stem}_task{ti}.md"
        seg_links = "\n".join(
            f"- [セグメント {s['index']}: {s['time']}]"
            f"({stem}_task{ti}_seg{s['index']}.md)"
            f" ({s['duration']} / {s['blocks']}blk)"
            for s in r["segments"]
        )
        task_md = (
            f"# Task {ti}: {r['keywords']}\n"
            f"*{r['duration']} / {r['blocks']}ブロック / "
            f"{len(r['segments'])}セグメント*\n\n"
            f"{r['summary']}\n\n"
            f"---\n\n"
            f"## セグメント詳細\n\n{seg_links}\n\n"
            f"---\n"
            f"[<- 統合レポートに戻る]({out_path.name})\n"
        )
        task_path.write_text(task_md, encoding="utf-8")
        task_files.append(task_path)

    # Layer 3: 統合レポート
    task_links = "\n".join(
        f"- [Task {r['index']}: {r['keywords']}]({stem}_task{r['index']}.md)"
        f" ({r['duration']} / {len(r['segments'])}seg)"
        for r in task_results
    )
    summary_md = f"{general_report}\n\n---\n\n## 個別タスクレポート\n\n{task_links}\n"
    out_path.write_text(summary_md, encoding="utf-8")

    print(f"\n  {C_GREEN}統合レポート: {out_path}{C_RESET}")
    for tf in task_files:
        print(f"  {C_DIM}  タスク: {tf}{C_RESET}")
    for sf in seg_files:
        print(f"  {C_DIM}    seg: {sf}{C_RESET}")
    print(
        f"  {C_DIM}(セグメント{total_segs}件 + タスクまとめ{len(tasks)}件"
        f" + 統合1件 = 計{total_segs + len(tasks) + 1}ファイル"
        f" / キャッシュ: {cache_hits}件"
        f" / API呼び出し: {api_calls}件){C_RESET}"
    )
