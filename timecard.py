"""claude-timecard: Claude Codeセッションログから労働時間を自動集計するCLI."""

import argparse
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

from lib.config import (
    JST, TimecardConfig,
    C_BOLD, C_DIM, C_GREEN, C_YELLOW, C_RESET,
)
from lib.parser.events import collect_events
from lib.analysis.blocks import build_blocks, format_duration, subdivide_blocks_by_keywords
from lib.analysis.tfidf import build_tfidf, _tokenize, _add_stopwords, add_synonym
from lib.analysis.tasks import infer_task_hierarchy, infer_tasks_branch_first
from lib.report.formatter import (
    print_daily_report,
    print_project_summary,
    print_branch_summary,
    print_keyword_distribution,
    print_task_report,
    print_json_output,
)
from lib.report.haiku import ai_classify_blocks, generate_report
from lib.visualization.plots import plot_keyword_graph, plot_keyword_ranking, plot_task_timeline, plot_branch_timeline
from lib.timing import Timer, enable_timing, get_timer


def parse_date(s: str) -> datetime:
    """YYYY-MM-DD or relative like '7d' を解析."""
    if s.endswith("d"):
        days = int(s[:-1])
        return datetime.now(JST).replace(
            hour=0, minute=0, second=0, microsecond=0
        ) - timedelta(days=days)
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=JST)


def main():
    parser = argparse.ArgumentParser(
        description="Claude Codeセッションログから労働時間を集計",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python timecard.py                        # 直近7日
  python timecard.py -d 14d                 # 直近14日
  python timecard.py -s 2026-02-21          # 2/21から今日まで
  python timecard.py -p myproject           # プロジェクトフィルタ
  python timecard.py --by-project           # プロジェクト別集計
  python timecard.py --by-branch            # ブランチ別集計
  python timecard.py --keywords             # キーワード頻度分布
  python timecard.py --tasks                # タスク推定（ブランチベース）
  python timecard.py --tasks --legacy-tasks # タスク推定（旧キーワードベース）
  python timecard.py --graph                # KDE曲線グラフ (PNG保存)
  python timecard.py --graph -o out.png     # 出力ファイル指定
  python timecard.py --ai                   # AIタスク分類 (claude -p)
  python timecard.py --ai --ai-model sonnet # AI分類をsonnetで実行
  python timecard.py --tasks --report       # タスク推定 + AIレポート生成
  python timecard.py --json                 # JSON出力
  python timecard.py --streams              # ストリーム別分析
  python timecard.py --calibration cal.json # キャリブレーション
        """,
    )
    parser.add_argument("-s", "--start", help="開始日 (YYYY-MM-DD)", default=None)
    parser.add_argument("-e", "--end", help="終了日 (YYYY-MM-DD)", default=None)
    parser.add_argument("-d", "--days", help="直近N日 (例: 7d, 14d)", default="7d")
    parser.add_argument(
        "-p", "--project", help="プロジェクト名フィルタ (部分一致)", default=None
    )
    parser.add_argument(
        "--idle", type=int, default=20, help="離席判定の閾値（分）default=20"
    )
    parser.add_argument(
        "--turn-time",
        type=int,
        default=5,
        help="1ターンあたりの読み時間（分）default=5",
    )
    parser.add_argument(
        "--by-project", action="store_true", help="プロジェクト別集計を表示"
    )
    parser.add_argument("--by-branch", action="store_true", help="ブランチ別集計を表示")
    parser.add_argument(
        "--keywords", action="store_true", help="キーワード頻度分布を表示"
    )
    parser.add_argument(
        "--tasks",
        action="store_true",
        help="タスク推定レポート（ブランチベース）",
    )
    parser.add_argument(
        "--legacy-tasks",
        action="store_true",
        help="旧キーワードベースのタスク推定を使用",
    )
    parser.add_argument(
        "--no-subdivide",
        action="store_true",
        help="キーワードシグナルによるブロック細分化を無効化",
    )
    parser.add_argument(
        "--graph", action="store_true", help="キーワード分布のKDE曲線グラフを出力 (PNG)"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="timecard_graph.png",
        help="グラフ出力ファイル名 (default: timecard_graph.png)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="グラフに表示するキーワード数 (default: 10)",
    )
    parser.add_argument(
        "--ai",
        action="store_true",
        help="AIでタスク分類（claude -p を使用）",
    )
    parser.add_argument(
        "--ai-model",
        default="haiku",
        help="AI分類に使うモデル (default: haiku)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="AIレポート生成 (--tasksと併用、claude -p でHaiku呼び出し)",
    )
    parser.add_argument(
        "--report-output",
        default="timecard_report.md",
        help="レポート出力ファイル名 (default: timecard_report.md)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="レポートキャッシュを無視して再生成",
    )
    parser.add_argument("--json", action="store_true", help="JSON形式で出力")
    parser.add_argument("--claude-dir", default=None, help="~/.claude/projects のパス")
    parser.add_argument(
        "--add-stop",
        nargs="+",
        metavar="WORD",
        help="ストップワードを追加 (stopwords.json に保存)",
    )
    # 新規フラグ
    parser.add_argument(
        "--calibration", default=None, help="キャリブレーションデータJSON"
    )
    parser.add_argument(
        "--timing", action="store_true", help="処理時間計測を表示"
    )
    parser.add_argument(
        "--streams", action="store_true", help="ストリーム別分析"
    )
    parser.add_argument(
        "--break-start", default=None, help="休憩開始時刻 (HH:MM)"
    )
    parser.add_argument(
        "--break-end", default=None, help="休憩終了時刻 (HH:MM)"
    )
    parser.add_argument(
        "--add-synonym", nargs="+", metavar=("CANONICAL", "ALIAS"),
        help="同義語を登録 (例: --add-synonym deploy デプロイ デプロイメント)"
    )

    args = parser.parse_args()

    # --add-stop: ストップワード追加して終了
    if args.add_stop:
        _add_stopwords(args.add_stop)
        return

    # --add-synonym: 同義語追加して終了
    if args.add_synonym:
        canonical = args.add_synonym[0]
        aliases = args.add_synonym[1:]
        if not aliases:
            print("エラー: 正規形の後に別名を1つ以上指定してください")
            return
        add_synonym(canonical, aliases)
        print(f"  同義語登録: {canonical} ← {', '.join(aliases)}")
        print(f"  保存先: ~/.config/claude-timecard/synonyms.json")
        return

    # タイミング計測
    timer: Timer | None = None
    if args.timing:
        timer = enable_timing()

    # Config生成
    config = TimecardConfig.from_args(args)

    # Claude projects dir
    if args.claude_dir:
        projects_dir = Path(args.claude_dir)
    else:
        projects_dir = Path.home() / ".claude" / "projects"

    if not projects_dir.exists():
        print(f"Error: {projects_dir} が見つかりません", file=sys.stderr)
        sys.exit(1)

    # 期間
    now = datetime.now(JST)
    if args.start:
        date_start = parse_date(args.start)
    else:
        date_start = parse_date(args.days)

    if args.end:
        date_end = parse_date(args.end) + timedelta(days=1)
    else:
        date_end = now + timedelta(days=1)

    # ヘッダー
    if not args.json:
        period = f"{date_start.strftime('%Y-%m-%d')} 〜 {now.strftime('%Y-%m-%d')}"
        proj_label = args.project or "全プロジェクト"
        print(f"\n{C_BOLD}{'═' * 70}{C_RESET}")
        print(
            f"  {C_BOLD}claude-timecard{C_RESET}"
            f"  {C_DIM}{period}  [{proj_label}]{C_RESET}"
        )
        print(
            f"  {C_DIM}idle閾値={args.idle}分  turn読み時間={args.turn_time}分{C_RESET}"
        )
        print(f"{C_BOLD}{'═' * 70}{C_RESET}")

    # データ収集
    if timer:
        _ctx_collect = timer.measure("collect_events")
        _ctx_collect.__enter__()
    events = collect_events(projects_dir, date_start, date_end, args.project)
    if timer:
        _ctx_collect.__exit__(None, None, None)

    if not events:
        if args.json:
            print("{}")
        else:
            print("\n  該当期間のセッションデータがありません。")
        sys.exit(0)

    # ブロック構築
    if timer:
        _ctx_blocks = timer.measure("build_blocks")
        _ctx_blocks.__enter__()
    blocks = build_blocks(events, args.idle, args.turn_time)
    if timer:
        _ctx_blocks.__exit__(None, None, None)

    # TF-IDF自動学習
    if timer:
        _ctx_tfidf = timer.measure("build_tfidf")
        _ctx_tfidf.__enter__()
    tfidf = build_tfidf(blocks)
    if timer:
        _ctx_tfidf.__exit__(None, None, None)

    # 自動ノイズ除去: 全ブロックの30%以上に出現する語を除外
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

    # キーワードシグナルによるブロック細分化
    if not args.no_subdivide and tfidf:
        blocks = subdivide_blocks_by_keywords(blocks, tfidf)

    # ストリーム別分析
    if args.streams:
        from lib.analysis.streams import group_by_stream, build_stream_blocks
        from lib.analysis.intervals import Interval, measure_union

        stream_blocks = build_stream_blocks(events, config)
        print(f"\n{C_BOLD}  ストリーム別分析{C_RESET}")
        print(f"  {'ストリーム':50} {'Active':>8} {'blocks':>7}")
        print(f"  {'-' * 70}")

        all_intervals: list[Interval] = []
        for (proj, branch), sblocks in sorted(stream_blocks.items()):
            active = sum(
                (b.end - b.start).total_seconds() / 60 for b in sblocks
            )
            label = f"{proj} : {branch}"
            print(
                f"  {label:50}"
                f" {format_duration(active):>8}"
                f" {len(sblocks):>6}"
            )
            for b in sblocks:
                all_intervals.append(
                    Interval(start=b.start, end=b.end, stream=(proj, branch))
                )

        union_min = measure_union(all_intervals)
        sum_min = sum(
            (b.end - b.start).total_seconds() / 60
            for sblocks in stream_blocks.values()
            for b in sblocks
        )
        print(f"\n  {C_BOLD}Union Active: {C_GREEN}{format_duration(union_min)}{C_RESET}")
        print(
            f"  {C_DIM}単純合計: {format_duration(sum_min)}"
            f"  重複: {format_duration(sum_min - union_min)}{C_RESET}"
        )

    # キャリブレーション
    cal_coeffs = None
    cal_locations = None
    if args.calibration:
        from lib.calibration.calibrator import CalibrationData, Calibrator
        from collections import defaultdict

        cal_data = CalibrationData.load(Path(args.calibration))
        calibrator = Calibrator(cal_data)

        # 日別Active時間を算出
        daily_active: dict[str, float] = defaultdict(float)
        for b in blocks:
            date_key = b.start.strftime("%Y-%m-%d")
            daily_active[date_key] += (b.end - b.start).total_seconds() / 60

        cal_coeffs = calibrator.compute_coefficients(dict(daily_active))

        # 場所情報をrecordから取得
        cal_locations = {}
        for record in cal_data.records:
            cal_locations[record.date] = record.location

        print(f"\n{C_BOLD}  キャリブレーション係数{C_RESET}")
        for loc, coeff in cal_coeffs.items():
            print(f"  {loc}: {coeff:.3f}")

    # 出力ディレクトリ（graph/report生成時）
    out_dir = None
    if args.graph or args.report:
        from datetime import datetime as _dt
        ts = _dt.now(JST).strftime("%Y%m%d_%H%M%S")
        out_dir = Path(__file__).parent / "output" / ts
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  {C_DIM}出力先: {out_dir}{C_RESET}")

    # 出力
    if args.json:
        print_json_output(events, blocks, tfidf=tfidf)
    else:
        print_daily_report(
            events, blocks, tfidf=tfidf,
            calibration_coeffs=cal_coeffs,
            calibration_locations=cal_locations,
        )
        if args.by_project:
            print_project_summary(events)
        if args.by_branch:
            print_branch_summary(events)
        if args.keywords:
            print_keyword_distribution(events, blocks, tfidf=tfidf)
        if args.tasks or args.report:
            if args.legacy_tasks:
                # 旧キーワードベースのタスク推定
                task_nodes, ctx_kws = infer_task_hierarchy(
                    blocks, tfidf=tfidf, top_n=args.top_n
                )
            else:
                # 新ブランチベースのタスク推定
                task_nodes, ctx_kws = infer_tasks_branch_first(
                    events, blocks, tfidf=tfidf,
                    idle_threshold=args.idle,
                    per_turn_time=args.turn_time,
                    top_n=args.top_n,
                )
            print_task_report(task_nodes, blocks, context_keywords=ctx_kws)
            if args.graph:
                plot_task_timeline(
                    task_nodes,
                    blocks,
                    context_keywords=ctx_kws,
                    output=str(out_dir / "task_timeline.png"),
                )
            if args.report:
                generate_report(
                    task_nodes,
                    blocks,
                    context_keywords=ctx_kws,
                    model=args.ai_model,
                    output=str(out_dir / "report.md"),
                    use_cache=not args.no_cache,
                )
        if args.graph:
            plot_keyword_graph(
                blocks, tfidf=tfidf, top_n=args.top_n,
                output=str(out_dir / "kde_density.png"),
            )
            plot_keyword_ranking(
                blocks, tfidf=tfidf, top_n=20,
                output=str(out_dir / "keyword_ranking.png"),
            )
            plot_branch_timeline(
                blocks, output=str(out_dir / "branch_timeline.png"),
            )
        if args.ai:
            ai_classify_blocks(blocks, model=args.ai_model)

    # タイミング表示
    if timer:
        summary = timer.summary()
        print(f"\n{C_BOLD}  Timing{C_RESET}")
        print(f"  {'-' * 50}")
        for step in summary["steps"]:
            print(f"  {step['label']:30} {step['seconds']:>8.3f}s")
        print(f"  {'─' * 50}")
        print(f"  {'Total':30} {summary['total_seconds']:>8.3f}s")


if __name__ == "__main__":
    main()
