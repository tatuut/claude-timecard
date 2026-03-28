"""claude-timecard MCP Server: Claude CodeからtimecardデータにアクセスするMCPサーバー.

MCP stdio transportはstdoutをJSON-RPCに使うため、
lib/内のprint()がstdoutに出るとプロトコルが壊れてハングする。
全ツール実行時にstdoutをstderrにリダイレクトして防止する。
"""

import contextlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# lib をインポートパスに追加
sys.path.insert(0, str(Path(__file__).parent))

from lib.config import TimecardConfig, JST, ATTRIBUTION_SHORT
from lib.parser.events import Event, collect_events, parse_timestamp
from lib.analysis.blocks import Block, build_blocks, subdivide_blocks_by_keywords
from lib.analysis.tfidf import build_tfidf, extract_keywords
from lib.analysis.streams import group_by_stream, build_stream_blocks
from lib.analysis.intervals import Interval, measure_union
from lib.analysis.tasks import infer_tasks_branch_first, infer_task_hierarchy
from lib.report.formatter import format_duration
from lib.timing import Timer

mcp = FastMCP("claude-timecard")

# --- helpers ---

_PROJECTS_DIR = Path.home() / ".claude" / "projects"
_META = {"tool": ATTRIBUTION_SHORT, "license": "AGPL-3.0"}


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
    timer: Timer | None = None,
) -> tuple[list[Event], list[Block], dict[str, float]]:
    """共通のデータロード処理."""
    if config is None:
        config = TimecardConfig()

    date_start = _parse_date(start)
    date_end = _parse_date(end) + timedelta(days=1) if end else datetime.now(JST) + timedelta(days=1)

    if timer:
        ctx = timer.measure("collect_events")
        ctx.__enter__()
    events = collect_events(_PROJECTS_DIR, date_start, date_end, project)
    if timer:
        ctx.__exit__(None, None, None)

    if not events:
        return [], [], {}

    if timer:
        ctx = timer.measure("build_blocks")
        ctx.__enter__()
    blocks = build_blocks(events, config.idle_threshold_min, config.per_turn_time_min)
    if timer:
        ctx.__exit__(None, None, None)

    if timer:
        ctx = timer.measure("build_tfidf")
        ctx.__enter__()
    tfidf = build_tfidf(blocks)
    if timer:
        ctx.__exit__(None, None, None)

    # ノイズ除去
    n_blocks = len(blocks)
    if n_blocks >= 4:
        from lib.analysis.tfidf import _tokenize
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

    # キーワード細分化
    if timer:
        ctx = timer.measure("subdivide_blocks")
        ctx.__enter__()
    blocks = subdivide_blocks_by_keywords(blocks, tfidf)
    if timer:
        ctx.__exit__(None, None, None)

    return events, blocks, tfidf


def _block_branch_summary(branches: list[str]) -> str:
    meaningful = [
        b for b in branches if b and b not in ("main", "master", "HEAD", "development")
    ]
    if not meaningful:
        return ""
    counts = Counter(meaningful)
    top = counts.most_common(1)
    return top[0][0].split("/")[-1] if top else ""


# --- MCP Tools ---
# 全ツールで contextlib.redirect_stdout(sys.stderr) を使い、
# lib/内のprint()がMCP transportを壊すのを防止する。


@mcp.tool()
def timecard_daily(
    start: str = "7d",
    end: str | None = None,
    project: str | None = None,
) -> str:
    """日別の作業時間レポートをJSON形式で返す。

    Args:
        start: 開始日 (YYYY-MM-DD) または相対日数 (例: "7d", "14d", "30d")
        end: 終了日 (YYYY-MM-DD)。省略時は今日まで
        project: プロジェクト名フィルタ (部分一致)。例: "myproject"
    """
    with contextlib.redirect_stdout(sys.stderr):
        timer = Timer()
        events, blocks, tfidf = _load_data(start, end, project, timer=timer)
        if not events:
            return json.dumps({"error": "該当期間のデータがありません"}, ensure_ascii=False)

        daily_blocks: dict[str, list[Block]] = defaultdict(list)
        for b in blocks:
            date_key = b.start.strftime("%Y-%m-%d")
            daily_blocks[date_key].append(b)

        result = {"days": {}, "summary": {}}
        grand_active = 0.0

        for date_key in sorted(daily_blocks.keys()):
            day_blocks = daily_blocks[date_key]
            active_min = sum((b.end - b.start).total_seconds() / 60 for b in day_blocks)
            grand_active += active_min

            result["days"][date_key] = {
                "active_hours": round(active_min / 60, 1),
                "active_minutes": round(active_min, 1),
                "turns": sum(b.turns for b in day_blocks),
                "blocks": len(day_blocks),
                "block_details": [
                    {
                        "time": f"{b.start.strftime('%H:%M')}~{b.end.strftime('%H:%M')}",
                        "duration_min": round((b.end - b.start).total_seconds() / 60),
                        "branch": _block_branch_summary(b.branches),
                        "keywords": [w for w, _ in extract_keywords(b.messages, 4, tfidf)],
                        "turns": b.turns,
                    }
                    for b in day_blocks
                ],
            }

        n_days = len(daily_blocks)
        result["summary"] = {
            "total_active_hours": round(grand_active / 60, 1),
            "working_days": n_days,
            "avg_daily_hours": round(grand_active / 60 / max(n_days, 1), 1),
            "total_messages": len(events),
        }
        result["timing"] = timer.summary()
        result["_meta"] = _META

    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def timecard_streams(
    start: str = "7d",
    end: str | None = None,
    project: str | None = None,
) -> str:
    """ストリーム（project×branch）別の作業時間分析をJSON形式で返す。

    各ブランチでの作業時間、union-of-intervalsによる重複除去後の合計を含む。

    Args:
        start: 開始日 (YYYY-MM-DD) または相対日数
        end: 終了日 (YYYY-MM-DD)
        project: プロジェクト名フィルタ
    """
    with contextlib.redirect_stdout(sys.stderr):
        events, blocks, tfidf = _load_data(start, end, project)
        if not events:
            return json.dumps({"error": "該当期間のデータがありません"}, ensure_ascii=False)

        config = TimecardConfig()
        stream_blocks = build_stream_blocks(events, config)

        streams = []
        all_intervals = []

        for (proj, branch), sblocks in sorted(stream_blocks.items()):
            active = sum((b.end - b.start).total_seconds() / 60 for b in sblocks)
            branch_short = branch.split("/")[-1] if "/" in branch else branch
            streams.append({
                "project": proj,
                "branch": branch_short,
                "branch_full": branch,
                "active_hours": round(active / 60, 1),
                "active_minutes": round(active, 1),
                "blocks": len(sblocks),
            })
            for b in sblocks:
                all_intervals.append(Interval(start=b.start, end=b.end, stream=(proj, branch)))

        union_min = measure_union(all_intervals)
        sum_min = sum(s["active_minutes"] for s in streams)

    return json.dumps({
        "_meta": _META,
        "streams": sorted(streams, key=lambda x: -x["active_minutes"]),
        "union_active_hours": round(union_min / 60, 1),
        "sum_active_hours": round(sum_min / 60, 1),
        "overlap_minutes": round(sum_min - union_min, 1),
    }, ensure_ascii=False, indent=2)


@mcp.tool()
def timecard_tasks(
    start: str = "7d",
    end: str | None = None,
    project: str | None = None,
    mode: str = "branch",
) -> str:
    """タスク推定結果をJSON形式で返す。

    ブランチ名とキーワード分析を組み合わせてタスクを自動推定する。

    Args:
        start: 開始日
        end: 終了日
        project: プロジェクト名フィルタ
        mode: "branch" (ブランチベース、デフォルト) or "keyword" (キーワードベース、旧方式)
    """
    with contextlib.redirect_stdout(sys.stderr):
        timer = Timer()
        events, blocks, tfidf = _load_data(start, end, project, timer=timer)
        if not events:
            return json.dumps({"error": "該当期間のデータがありません"}, ensure_ascii=False)

        config = TimecardConfig()

        with timer.measure("infer_tasks"):
            if mode == "branch":
                task_nodes, ctx_kws = infer_tasks_branch_first(
                    events, blocks, tfidf=tfidf,
                    idle_threshold=config.idle_threshold_min,
                    per_turn_time=config.per_turn_time_min,
                )
            else:
                task_nodes, ctx_kws = infer_task_hierarchy(blocks, tfidf=tfidf)

        tasks = []
        for t in task_nodes:
            tasks.append({
                "keywords": t.keywords,
                "branches": t.branches if hasattr(t, "branches") else [],
                "active_hours": round(t.active_minutes / 60, 1),
                "active_minutes": round(t.active_minutes, 1),
                "sigma_minutes": round(t.sigma_minutes, 1),
                "blocks": len(t.block_set),
                "peak_times": t.peak_times[:5],
                "children": [
                    {
                        "keywords": c.keywords,
                        "active_hours": round(c.active_minutes / 60, 1),
                        "relationship": c.relationship,
                    }
                    for c in t.children
                ],
            })

        all_blocks_union = set()
        for t in task_nodes:
            all_blocks_union |= t.block_set
        union_min = sum(
            (blocks[bi].end - blocks[bi].start).total_seconds() / 60
            for bi in all_blocks_union
            if bi < len(blocks)
        )

    return json.dumps({
        "_meta": _META,
        "tasks": tasks,
        "context_keywords": ctx_kws,
        "union_active_hours": round(union_min / 60, 1),
        "mode": mode,
        "timing": timer.summary(),
    }, ensure_ascii=False, indent=2)


@mcp.tool()
def timecard_graph(
    start: str = "7d",
    end: str | None = None,
    project: str | None = None,
    output_dir: str | None = None,
) -> str:
    """KDE密度曲線・タスクタイムライン・ブランチタイムライン・キーワードランキングのPNGグラフを生成する。

    output/{YYYYMMDD_HHMMSS}/ ディレクトリに出力。

    Args:
        start: 開始日
        end: 終了日
        project: プロジェクト名フィルタ
        output_dir: 出力ベースディレクトリ。省略時はtimecardプロジェクトのoutput/
    """
    with contextlib.redirect_stdout(sys.stderr):
        events, blocks, tfidf = _load_data(start, end, project)
        if not events:
            return json.dumps({"error": "該当期間のデータがありません"}, ensure_ascii=False)

        ts = datetime.now(JST).strftime("%Y%m%d_%H%M%S")
        if output_dir:
            out_dir = Path(output_dir) / ts
        else:
            out_dir = Path(__file__).parent / "output" / ts
        out_dir.mkdir(parents=True, exist_ok=True)

        config = TimecardConfig()
        generated = []

        # 1. KDE密度曲線
        try:
            from lib.visualization.plots import plot_keyword_graph
            p = str(out_dir / "kde_density.png")
            plot_keyword_graph(blocks, tfidf=tfidf, top_n=10, output=p)
            generated.append({"type": "kde_density", "path": p})
        except Exception as e:
            generated.append({"type": "kde_density", "error": str(e)})

        # 2. キーワードランキング
        try:
            from lib.visualization.plots import plot_keyword_ranking
            p = str(out_dir / "keyword_ranking.png")
            plot_keyword_ranking(blocks, tfidf=tfidf, top_n=20, output=p)
            generated.append({"type": "keyword_ranking", "path": p})
        except Exception as e:
            generated.append({"type": "keyword_ranking", "error": str(e)})

        # 3. タスクタイムライン
        try:
            from lib.visualization.plots import plot_task_timeline
            task_nodes, ctx_kws = infer_tasks_branch_first(
                events, blocks, tfidf=tfidf,
                idle_threshold=config.idle_threshold_min,
                per_turn_time=config.per_turn_time_min,
            )
            p = str(out_dir / "task_timeline.png")
            plot_task_timeline(task_nodes, blocks, context_keywords=ctx_kws, output=p)
            generated.append({"type": "task_timeline", "path": p})
        except Exception as e:
            generated.append({"type": "task_timeline", "error": str(e)})

        # 4. ブランチタイムライン
        try:
            from lib.visualization.plots import plot_branch_timeline
            stream_blocks = build_stream_blocks(events, config)
            p = str(out_dir / "branch_timeline.png")
            plot_branch_timeline(stream_blocks, output=p)
            generated.append({"type": "branch_timeline", "path": p})
        except Exception as e:
            generated.append({"type": "branch_timeline", "error": str(e)})

    return json.dumps({"_meta": _META, "generated": generated}, ensure_ascii=False, indent=2)


@mcp.tool()
def timecard_keywords(
    start: str = "7d",
    end: str | None = None,
    project: str | None = None,
    top_n: int = 20,
) -> str:
    """期間中のキーワード頻度分析をJSON形式で返す。

    TF-IDFスコアで重み付けされた上位キーワードと出現ブロック数を含む。

    Args:
        start: 開始日
        end: 終了日
        project: プロジェクト名フィルタ
        top_n: 返すキーワード数
    """
    with contextlib.redirect_stdout(sys.stderr):
        events, blocks, tfidf = _load_data(start, end, project)
        if not events:
            return json.dumps({"error": "該当期間のデータがありません"}, ensure_ascii=False)

        from lib.analysis.tfidf import _tokenize

        block_word_counts = []
        for b in blocks:
            wc = Counter()
            for msg in b.messages:
                for token in _tokenize(msg):
                    wc[token] += 1
            block_word_counts.append(wc)

        word_df = Counter()
        for wc in block_word_counts:
            for w in wc:
                word_df[w] += 1

        total_tf = Counter()
        for wc in block_word_counts:
            total_tf.update(wc)

        ranked = sorted(tfidf.items(), key=lambda x: -x[1])[:top_n]

        keywords = []
        for word, score in ranked:
            keywords.append({
                "word": word,
                "tfidf_score": round(score, 2),
                "total_count": total_tf.get(word, 0),
                "block_count": word_df.get(word, 0),
                "total_blocks": len(blocks),
            })

    return json.dumps({"_meta": _META, "keywords": keywords}, ensure_ascii=False, indent=2)


@mcp.tool()
def timecard_compare_months(
    month1: str,
    month2: str,
    project: str | None = None,
) -> str:
    """2つの月の作業時間を比較する。

    Args:
        month1: 比較元の月 (YYYY-MM 形式、例: "2026-02")
        month2: 比較先の月 (YYYY-MM 形式、例: "2026-03")
        project: プロジェクト名フィルタ
    """
    with contextlib.redirect_stdout(sys.stderr):
        results = {}
        for label, month in [("month1", month1), ("month2", month2)]:
            year, mon = month.split("-")
            start = f"{year}-{mon}-01"
            if int(mon) == 12:
                end_date = f"{int(year)+1}-01-01"
            else:
                end_date = f"{year}-{int(mon)+1:02d}-01"

            events, blocks, tfidf = _load_data(start, end_date, project)

            daily_blocks: dict[str, float] = defaultdict(float)
            for b in blocks:
                date_key = b.start.strftime("%Y-%m-%d")
                daily_blocks[date_key] += (b.end - b.start).total_seconds() / 60

            total_min = sum(daily_blocks.values())
            results[label] = {
                "month": month,
                "total_active_hours": round(total_min / 60, 1),
                "working_days": len(daily_blocks),
                "avg_daily_hours": round(total_min / 60 / max(len(daily_blocks), 1), 1),
                "total_messages": len(events),
                "daily": {
                    k: round(v / 60, 1) for k, v in sorted(daily_blocks.items())
                },
            }

    results["_meta"] = _META
    return json.dumps(results, ensure_ascii=False, indent=2)


@mcp.tool()
def timecard_report(
    start: str = "7d",
    end: str | None = None,
    project: str | None = None,
    model: str = "haiku",
    depth: str = "full",
    output_dir: str | None = None,
) -> str:
    """AIレポートを生成する。depthで生成量を制御可能。

    Args:
        start: 開始日
        end: 終了日
        project: プロジェクト名フィルタ
        model: AIモデル (default: "haiku")
        depth: レポート深度。"full"=全3階層, "tasks-only"=タスクまとめのみ(セグメント省略), "structure-only"=AI不使用(構造分析のみ)
        output_dir: Markdown出力先ディレクトリ
    """
    with contextlib.redirect_stdout(sys.stderr):
        events, blocks, tfidf = _load_data(start, end, project)
        if not events:
            return json.dumps({"error": "該当期間のデータがありません"}, ensure_ascii=False)

        config = TimecardConfig()

        task_nodes, ctx_kws = infer_tasks_branch_first(
            events, blocks, tfidf=tfidf,
            idle_threshold=config.idle_threshold_min,
            per_turn_time=config.per_turn_time_min,
        )

        if not task_nodes:
            return json.dumps({"error": "タスクを推定できませんでした"}, ensure_ascii=False)

        ts = datetime.now(JST).strftime("%Y%m%d_%H%M%S")
        if output_dir:
            out_dir = Path(output_dir) / ts
        else:
            out_dir = Path(__file__).parent / "output" / ts
        out_dir.mkdir(parents=True, exist_ok=True)

        # structure-only: AIを使わず構造だけ返す
        if depth == "structure-only":
            structure = []
            for ti, task in enumerate(task_nodes, 1):
                from lib.report.haiku import _segment_task_blocks
                segments = _segment_task_blocks(task, blocks)
                structure.append({
                    "task_index": ti,
                    "keywords": task.keywords,
                    "branches": task.branches if hasattr(task, "branches") else [],
                    "active_hours": round(task.active_minutes / 60, 1),
                    "segments": len(segments),
                    "blocks": len(task.block_set),
                })
            return json.dumps({
                "depth": "structure-only",
                "tasks": structure,
                "context_keywords": ctx_kws,
                "note": "AI未使用。構造分析のみ。",
            }, ensure_ascii=False, indent=2)

        output_path = str(out_dir / "report.md")

        from lib.report.haiku import generate_report
        generate_report(
            task_nodes,
            blocks,
            context_keywords=ctx_kws,
            model=model,
            output=output_path,
            use_cache=True,
        )

        stem = Path(output_path).stem
        generated_files = [output_path]
        for ti, task in enumerate(task_nodes, 1):
            task_path = out_dir / f"{stem}_task{ti}.md"
            if task_path.exists():
                generated_files.append(str(task_path))

    return json.dumps({
        "_meta": _META,
        "depth": depth,
        "report_path": output_path,
        "generated_files": generated_files,
        "tasks": len(task_nodes),
        "model": model,
    }, ensure_ascii=False, indent=2)


@mcp.tool()
def timecard_detail(
    start: str = "7d",
    end: str | None = None,
    project: str | None = None,
    date: str | None = None,
) -> str:
    """最も細かい粒度のブロック分析を返す。キーワード細分化後の全ブロック情報を含む。

    各ブロックのメッセージ数、キーワード、ブランチ、時間帯を返す。
    dateを指定すると特定の日のみ。

    Args:
        start: 開始日
        end: 終了日
        project: プロジェクト名フィルタ
        date: 特定日のみ取得 (YYYY-MM-DD)
    """
    with contextlib.redirect_stdout(sys.stderr):
        timer = Timer()
        events, blocks, tfidf = _load_data(start, end, project, timer=timer)
        if not events:
            return json.dumps({"error": "該当期間のデータがありません"}, ensure_ascii=False)

        if date:
            blocks = [b for b in blocks if b.start.strftime("%Y-%m-%d") == date]

        block_details = []
        for b in blocks:
            kw = extract_keywords(b.messages, top_n=6, tfidf_scores=tfidf)
            branch = _block_branch_summary(b.branches)
            dur_min = (b.end - b.start).total_seconds() / 60

            block_details.append({
                "date": b.start.strftime("%Y-%m-%d"),
                "start": b.start.strftime("%H:%M"),
                "end": b.end.strftime("%H:%M"),
                "duration_minutes": round(dur_min, 1),
                "branch": branch,
                "projects": sorted(b.projects),
                "keywords": [{"word": w, "count": c} for w, c in kw],
                "turns": b.turns,
                "message_count": len(b.messages),
                "pr_numbers": sorted(set(b.pr_numbers)),
            })

    return json.dumps({
        "_meta": _META,
        "total_blocks": len(block_details),
        "total_minutes": round(sum(b["duration_minutes"] for b in block_details), 1),
        "blocks": block_details,
        "timing": timer.summary(),
    }, ensure_ascii=False, indent=2)


@mcp.tool()
def timecard_daily_report(
    start: str = "7d",
    end: str | None = None,
    project: str | None = None,
    report_dir: str | None = None,
) -> str:
    """日別Active時間にAIレポートのセグメント要約を紐付けて返す。

    timecard_report で生成済みのレポートファイルがある場合、各日のブロックに対応する
    セグメントの内容を読み込んで付与する。レポートがない場合はブランチ名+キーワードのみ。

    タイムシート作成の「作業内容」欄を書くために使う。

    Args:
        start: 開始日
        end: 終了日
        project: プロジェクト名フィルタ
        report_dir: レポートが格納されたディレクトリ (output/YYYYMMDD_HHMMSS)。省略時は最新を自動検索
    """
    with contextlib.redirect_stdout(sys.stderr):
        events, blocks, tfidf = _load_data(start, end, project)
        if not events:
            return json.dumps({"error": "該当期間のデータがありません"}, ensure_ascii=False)

        config = TimecardConfig()

        # タスク推定
        task_nodes, ctx_kws = infer_tasks_branch_first(
            events, blocks, tfidf=tfidf,
            idle_threshold=config.idle_threshold_min,
            per_turn_time=config.per_turn_time_min,
        )

        # セグメント → 日付マッピングを構築
        from lib.report.haiku import _segment_task_blocks
        seg_map: list[dict] = []  # [{task_index, seg_index, dates, block_idxs, keywords, branches}]

        for ti, task in enumerate(task_nodes, 1):
            segments = _segment_task_blocks(task, blocks)
            for si, seg_idxs in enumerate(segments, 1):
                seg_dates = sorted({blocks[bi].start.strftime("%Y-%m-%d") for bi in seg_idxs})
                seg_start = blocks[seg_idxs[0]].start
                seg_end = blocks[seg_idxs[-1]].end
                seg_dur = sum(
                    (blocks[bi].end - blocks[bi].start).total_seconds() / 60
                    for bi in seg_idxs
                )
                seg_map.append({
                    "task_index": ti,
                    "seg_index": si,
                    "keywords": task.keywords[:5],
                    "branches": task.branches[:3] if hasattr(task, "branches") else [],
                    "dates": seg_dates,
                    "time_range": f"{seg_start.strftime('%m/%d %H:%M')}~{seg_end.strftime('%m/%d %H:%M')}",
                    "duration_minutes": round(seg_dur, 1),
                    "block_count": len(seg_idxs),
                })

        # レポートファイルの検索・読み込み
        report_contents: dict[str, str] = {}  # "task{ti}_seg{si}" → content
        if report_dir:
            rdir = Path(report_dir)
        else:
            # output/ 配下の最新ディレクトリを探す
            output_base = Path(__file__).parent / "output"
            if output_base.exists():
                subdirs = sorted(output_base.iterdir(), reverse=True)
                rdir = subdirs[0] if subdirs else None
            else:
                rdir = None

        if rdir and rdir.exists():
            for md_file in rdir.glob("*_task*_seg*.md"):
                report_contents[md_file.stem.split("_", 1)[1]] = md_file.read_text(encoding="utf-8")
            # タスクまとめファイルも読む
            for md_file in rdir.glob("*_task[0-9]*.md"):
                if "_seg" not in md_file.name:
                    key = md_file.stem.split("_", 1)[1]
                    report_contents[key] = md_file.read_text(encoding="utf-8")

        # セグメントにレポート内容を付与
        for seg in seg_map:
            ti, si = seg["task_index"], seg["seg_index"]
            seg_key = f"task{ti}_seg{si}"
            task_key = f"task{ti}"
            seg["report_segment"] = report_contents.get(seg_key, None)
            seg["report_task_summary"] = report_contents.get(task_key, None)

        # 日別にグループ化
        daily: dict[str, dict] = {}
        for b in blocks:
            date_key = b.start.strftime("%Y-%m-%d")
            if date_key not in daily:
                daily[date_key] = {"active_minutes": 0.0, "blocks": 0, "segments": []}
            daily[date_key]["active_minutes"] += (b.end - b.start).total_seconds() / 60
            daily[date_key]["blocks"] += 1

        for seg in seg_map:
            for date in seg["dates"]:
                if date in daily:
                    daily[date]["segments"].append({
                        "task_index": seg["task_index"],
                        "seg_index": seg["seg_index"],
                        "keywords": seg["keywords"],
                        "branches": seg["branches"],
                        "time_range": seg["time_range"],
                        "duration_minutes": seg["duration_minutes"],
                        "has_report": seg["report_segment"] is not None,
                        "report_excerpt": (
                            seg["report_segment"][:500] if seg["report_segment"] else None
                        ),
                    })

        # 日付ソートして返す
        result_days = {}
        for date in sorted(daily.keys()):
            d = daily[date]
            result_days[date] = {
                "active_hours": round(d["active_minutes"] / 60, 1),
                "active_minutes": round(d["active_minutes"], 1),
                "blocks": d["blocks"],
                "segments": d["segments"],
            }

    return json.dumps({
        "_meta": _META,
        "days": result_days,
        "total_segments": len(seg_map),
        "report_dir": str(rdir) if rdir else None,
        "report_files_found": len(report_contents),
    }, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    mcp.run(transport="stdio")
