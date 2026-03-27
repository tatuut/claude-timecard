"""ANSI出力, print_daily_report, print_project_summary, etc."""

import json
import sys
from collections import Counter, defaultdict

from ..config import (
    C_BOLD, C_BG_BLUE, C_BLUE, C_CYAN, C_DIM, C_GREEN,
    C_MAGENTA, C_RED, C_RESET, C_YELLOW,
)
from ..parser.events import Event
from ..analysis.blocks import Block, build_blocks, calc_gross_time, format_duration
from ..analysis.tfidf import extract_keywords, _tokenize


def _format_branch(branch: str) -> str:
    """ブランチ名を短く表示."""
    if not branch or branch in ("main", "master", "HEAD", "development"):
        return ""
    parts = branch.split("/")
    if len(parts) >= 3:
        return "/".join(parts[2:])
    if len(parts) == 2:
        return parts[1]
    return branch


def _block_branch_summary(branches: list[str]) -> str:
    """ブロック内の代表ブランチを返す."""
    meaningful = [
        b for b in branches if b and b not in ("main", "master", "HEAD", "development")
    ]
    if not meaningful:
        return ""
    counts = Counter(meaningful)
    top = counts.most_common(2)
    result = _format_branch(top[0][0])
    if len(top) > 1:
        result += f" +{len(top) - 1}"
    return result


def _block_pr_summary(pr_numbers: list[int]) -> str:
    """ブロック内のPR番号を返す."""
    unique = sorted(set(pr_numbers))
    if not unique:
        return ""
    if len(unique) <= 3:
        return " ".join(f"#{n}" for n in unique)
    return " ".join(f"#{n}" for n in unique[:3]) + f" +{len(unique) - 3}"


def _format_keywords(keywords: list[tuple[str, int]]) -> str:
    """キーワードリストを表示用文字列に."""
    if not keywords:
        return ""
    parts = []
    for word, count in keywords:
        if count > 1:
            parts.append(f"{word}({count})")
        else:
            parts.append(word)
    return " ".join(parts)


def print_daily_report(
    events: list[Event],
    blocks: list[Block],
    tfidf: dict[str, float] | None = None,
    calibration_coeffs: dict[str, float] | None = None,
    calibration_locations: dict[str, str] | None = None,
):
    """日別レポートを表示."""
    daily_blocks: dict[str, list[Block]] = defaultdict(list)
    for b in blocks:
        date_key = b.start.strftime("%m/%d (%a)")
        daily_blocks[date_key].append(b)

    gross = calc_gross_time(events)

    grand_active = 0.0
    grand_gross = 0.0
    grand_estimated = 0.0
    has_calibration = calibration_coeffs is not None
    cal_locations = calibration_locations or {}

    # 日付キーからYYYY-MM-DDを抽出するためのマッピング
    daily_date_map: dict[str, str] = {}
    for b in blocks:
        date_key = b.start.strftime("%m/%d (%a)")
        if date_key not in daily_date_map:
            daily_date_map[date_key] = b.start.strftime("%Y-%m-%d")

    for date_key in sorted(daily_blocks.keys()):
        day_blocks = daily_blocks[date_key]
        active_min = sum((b.end - b.start).total_seconds() / 60 for b in day_blocks)
        grand_active += active_min

        gross_start, gross_end = gross.get(date_key, (None, None))
        gross_min = 0.0
        if gross_start and gross_end:
            gross_min = (gross_end - gross_start).total_seconds() / 60 + 5
            grand_gross += gross_min

        day_turns = sum(b.turns for b in day_blocks)
        bar = f"{C_GREEN}{'█' * int(active_min / 60 * 3)}{C_RESET}"

        # キャリブレーション推定
        cal_str = ""
        if has_calibration:
            iso_date = daily_date_map.get(date_key, "")
            location = cal_locations.get(iso_date, "unknown")
            coeff = calibration_coeffs.get(
                location, calibration_coeffs.get("default", 1.0)
            )
            estimated = active_min * coeff
            grand_estimated += estimated
            cal_str = (
                f"  {C_MAGENTA}推定: {format_duration(estimated)}{C_RESET}"
            )

        print(
            f"\n{C_BOLD}{C_BG_BLUE} {date_key} {C_RESET}"
            f"  Active: {C_GREEN}{format_duration(active_min)}{C_RESET}"
            f"  Gross: {C_YELLOW}{format_duration(gross_min)}{C_RESET}"
            f"{cal_str}"
            f"  {C_DIM}{day_turns} turns{C_RESET}  {bar}"
        )

        for b in day_blocks:
            dur = (b.end - b.start).total_seconds() / 60
            proj_str = ", ".join(sorted(b.projects))

            branch_str = _block_branch_summary(b.branches)
            branch_display = f"  {C_BLUE}⎇ {branch_str}{C_RESET}" if branch_str else ""

            pr_str = _block_pr_summary(b.pr_numbers)
            pr_display = f"  {C_RED}{pr_str}{C_RESET}" if pr_str else ""

            kw = extract_keywords(b.messages, top_n=4, tfidf_scores=tfidf)
            kw_str = _format_keywords(kw)
            kw_display = f"  {C_MAGENTA}{kw_str}{C_RESET}" if kw_str else ""

            print(
                f"  {C_CYAN}{b.start.strftime('%H:%M')}"
                f"〜{b.end.strftime('%H:%M')}{C_RESET}"
                f" ({dur:.0f}m) {C_DIM}{b.turns}t{C_RESET}"
                f"  {proj_str}{branch_display}{pr_display}"
            )
            if kw_display:
                print(f"    {kw_display}")

    # サマリー
    n_days = len(daily_blocks)
    print(f"\n{'═' * 70}")
    est_line = ""
    if has_calibration:
        est_line = (
            f"  {C_BOLD}推定合計: {C_MAGENTA}"
            f"{format_duration(grand_estimated)}{C_RESET}"
        )
    print(
        f"  {C_BOLD}合計 Active: {C_GREEN}{format_duration(grand_active)}{C_RESET}"
        f"  {C_BOLD}Gross: {C_YELLOW}{format_duration(grand_gross)}{C_RESET}"
    )
    if est_line:
        print(est_line)
    print(f"  メッセージ: {len(events)}  稼働日数: {n_days}日")
    if n_days:
        print(
            f"  1日平均 Active: {C_GREEN}"
            f"{format_duration(grand_active / n_days)}{C_RESET}"
            f"  Gross: {C_YELLOW}"
            f"{format_duration(grand_gross / n_days)}{C_RESET}"
        )
    print(f"{'═' * 70}")


def print_project_summary(events: list[Event]):
    """プロジェクト別の集計."""
    proj_events: dict[str, list[Event]] = defaultdict(list)
    for ev in events:
        proj_events[ev.project].append(ev)

    print(f"\n{C_BOLD}  プロジェクト別集計{C_RESET}")
    print(f"  {'プロジェクト':35} {'Active':>8} {'turns':>7}")
    print(f"  {'-' * 55}")

    rows = []
    for proj, pevents in proj_events.items():
        pblocks = build_blocks(pevents)
        total = sum((b.end - b.start).total_seconds() / 60 for b in pblocks)
        rows.append((proj, total, len(pevents)))

    rows.sort(key=lambda x: -x[1])
    for proj, total, turns in rows:
        bar = f"{C_GREEN}{'█' * int(total / 60)}{C_RESET}"
        print(f"  {proj:35} {format_duration(total):>8} {turns:>6}t  {bar}")


def print_branch_summary(events: list[Event]):
    """ブランチ別の集計."""
    branch_events: dict[str, list[Event]] = defaultdict(list)
    for ev in events:
        if ev.branch and ev.branch not in (
            "main",
            "master",
            "HEAD",
            "development",
        ):
            branch_events[ev.branch].append(ev)

    if not branch_events:
        print(f"\n{C_DIM}  ブランチ情報なし{C_RESET}")
        return

    print(f"\n{C_BOLD}  ブランチ別集計{C_RESET}")
    print(f"  {'ブランチ':45} {'Active':>8} {'turns':>7} {'プロジェクト'}")
    print(f"  {'-' * 80}")

    rows = []
    for branch, bevents in branch_events.items():
        bblocks = build_blocks(bevents)
        total = sum((b.end - b.start).total_seconds() / 60 for b in bblocks)
        projs = sorted({ev.project for ev in bevents})
        rows.append((branch, total, len(bevents), projs))

    rows.sort(key=lambda x: -x[1])
    for branch, total, turns, projs in rows[:15]:
        short_branch = _format_branch(branch) or branch
        proj_str = ", ".join(projs)
        bar = f"{C_GREEN}{'█' * max(1, int(total / 60))}{C_RESET}"
        print(
            f"  {C_BLUE}{short_branch:45}{C_RESET}"
            f" {format_duration(total):>8} {turns:>6}t  {proj_str}  {bar}"
        )


def print_keyword_distribution(
    events: list[Event],
    blocks: list[Block],
    tfidf: dict[str, float] | None = None,
):
    """キーワード頻度の時間分布を表示."""
    if not blocks:
        return

    block_word_counts: list[Counter] = []
    for b in blocks:
        wc: Counter = Counter()
        for msg in b.messages:
            for token in _tokenize(msg):
                wc[token] += 1
        block_word_counts.append(wc)

    if tfidf:
        ranked = sorted(tfidf.items(), key=lambda x: -x[1])[:20]
        display_words = [w for w, _ in ranked]
    else:
        total_counts: Counter = Counter()
        for wc in block_word_counts:
            total_counts.update(wc)
        display_words = [w for w, _ in total_counts.most_common(20)]

    total_tf: Counter = Counter()
    for wc in block_word_counts:
        total_tf.update(wc)

    print(f"\n{C_BOLD}  キーワード頻度分布（上位20）{C_RESET}")
    print(f"  {'キーワード':20} {'出現':>5} {'時間帯分布'}")
    print(f"  {'-' * 70}")

    t_min = min(b.start for b in blocks)
    t_max = max(b.end for b in blocks)
    total_span = (t_max - t_min).total_seconds()
    if total_span == 0:
        return

    bar_width = 40
    for word in display_words:
        count = total_tf.get(word, 0)
        bar = [" "] * bar_width
        for i, b in enumerate(blocks):
            wc = block_word_counts[i].get(word, 0)
            if wc > 0:
                pos = (b.start - t_min).total_seconds() / total_span
                idx = min(int(pos * bar_width), bar_width - 1)
                bar[idx] = "█" if wc >= 5 else "▓" if wc >= 3 else "░"

        bar_str = "".join(bar)
        print(
            f"  {C_MAGENTA}{word:20}{C_RESET}"
            f" {count:>5}"
            f"  {C_DIM}|{C_RESET}{C_CYAN}{bar_str}{C_RESET}{C_DIM}|{C_RESET}"
        )

    label_start = t_min.strftime("%m/%d")
    label_end = t_max.strftime("%m/%d")
    print(
        f"  {'':20} {'':>5}  {C_DIM} {label_start:>{bar_width // 2}}{label_end:>{bar_width // 2}} {C_RESET}"
    )


def print_task_report(
    tasks: list,
    blocks: list[Block],
    context_keywords: list[str] | None = None,
):
    """タスク推定レポートを表示."""
    if not tasks:
        print(f"\n  {C_DIM}タスクを推定できませんでした{C_RESET}")
        return

    print(f"\n{C_BOLD}{'═' * 70}{C_RESET}")
    print(f"  {C_BOLD}タスク推定レポート{C_RESET}")
    print(f"{C_BOLD}{'═' * 70}{C_RESET}")

    if context_keywords:
        ctx_str = ", ".join(context_keywords[:8])
        print(f"\n  {C_DIM}文脈（全体に広がる語）: {ctx_str}{C_RESET}")

    task_contexts: dict[int, list[str]] = {}
    if context_keywords:
        blk_vocab: list[set[str]] = []
        for b in blocks:
            vocab: set[str] = set()
            for msg in b.messages:
                vocab.update(_tokenize(msg))
            blk_vocab.append(vocab)

        for ti, task in enumerate(tasks):
            ctx_hits: Counter = Counter()
            for bi in task.block_set:
                for cw in context_keywords:
                    if cw in blk_vocab[bi]:
                        ctx_hits[cw] += 1
            n_blk = max(len(task.block_set), 1)
            task_contexts[ti] = [
                w for w, c in ctx_hits.most_common() if c / n_blk >= 0.3
            ][:4]

    all_blocks_union: set[int] = set()

    for i, task in enumerate(tasks, 1):
        kw_str = ", ".join(task.keywords)
        time_str = format_duration(task.active_minutes)
        sig_str = format_duration(task.sigma_minutes)
        peaks = ", ".join(task.peak_times[:4])
        n_blk = len(task.block_set)

        print(
            f"\n  {C_BOLD}{C_CYAN}■ Task {i}: {kw_str}{C_RESET}"
            f"  {C_GREEN}{time_str} ± {sig_str}{C_RESET}"
            f"  {C_DIM}({n_blk}blk){C_RESET}"
        )
        ctx = task_contexts.get(i - 1, [])
        if ctx:
            print(f"    {C_DIM}文脈: {', '.join(ctx)}{C_RESET}")
        if peaks:
            print(f"    {C_DIM}ピーク: {peaks}{C_RESET}")

        all_blocks_union |= task.block_set

        for j, child in enumerate(task.children):
            ckw = ", ".join(child.keywords)
            ct = format_duration(child.active_minutes)
            cs = format_duration(child.sigma_minutes)
            rel_label = "含有" if child.relationship == "subtask" else "部分"
            rel_color = C_YELLOW if child.relationship == "subtask" else C_MAGENTA
            is_last = j == len(task.children) - 1
            connector = "└──" if is_last else "├──"
            cpks = ", ".join(child.peak_times[:3])

            print(
                f"    {C_DIM}{connector}{C_RESET}"
                f" {C_MAGENTA}{ckw}{C_RESET}"
                f"  {rel_color}({rel_label}){C_RESET}"
                f"  {C_GREEN}{ct} ± {cs}{C_RESET}"
            )
            if cpks:
                prefix = "     " if is_last else "│    "
                print(f"    {C_DIM}{prefix}ピーク: {cpks}{C_RESET}")

    union_min = sum(
        (blocks[bi].end - blocks[bi].start).total_seconds() / 60
        for bi in all_blocks_union
    )
    sum_indiv = sum(t.active_minutes for t in tasks)
    overlap_pct = (sum_indiv - union_min) / sum_indiv * 100 if sum_indiv > 0 else 0

    print(f"\n  {'─' * 60}")
    print(
        f"  {C_BOLD}合計（和集合）: {C_GREEN}"
        f"{format_duration(union_min)}{C_RESET}"
        f"  {C_DIM}({len(all_blocks_union)}blk){C_RESET}"
    )
    print(
        f"  {C_DIM}単純合計: {format_duration(sum_indiv)}"
        f"  重複率: {overlap_pct:.0f}%{C_RESET}"
    )
    print(f"  {C_DIM}※サブタスクのブロック ⊆ 親タスクのブロック{C_RESET}")
    print(f"{'═' * 70}")


def print_json_output(
    events: list[Event], blocks: list[Block], tfidf: dict[str, float] | None = None
):
    """JSON形式で出力."""
    output: dict = {
        "total_events": len(events),
        "total_blocks": len(blocks),
        "days": {},
    }

    daily_blocks: dict[str, list[Block]] = defaultdict(list)
    for b in blocks:
        date_key = b.start.strftime("%Y-%m-%d")
        daily_blocks[date_key].append(b)

    for date_key in sorted(daily_blocks.keys()):
        day_blocks = daily_blocks[date_key]
        active_min = sum((b.end - b.start).total_seconds() / 60 for b in day_blocks)
        output["days"][date_key] = {
            "active_minutes": round(active_min, 1),
            "turns": sum(b.turns for b in day_blocks),
            "blocks": [
                {
                    "start": b.start.isoformat(),
                    "end": b.end.isoformat(),
                    "duration_minutes": round(
                        (b.end - b.start).total_seconds() / 60, 1
                    ),
                    "projects": sorted(b.projects),
                    "branch": _block_branch_summary(b.branches),
                    "pr_numbers": sorted(set(b.pr_numbers)),
                    "keywords": [
                        w
                        for w, _ in extract_keywords(b.messages, 5, tfidf_scores=tfidf)
                    ],
                    "turns": b.turns,
                }
                for b in day_blocks
            ],
        }

    json.dump(output, sys.stdout, ensure_ascii=False, indent=2)
    print()
