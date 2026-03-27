"""Block dataclass, build_blocks, subdivide_blocks_by_keywords, SubBlock, subdivide_block_soft."""

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ..parser.events import Event


@dataclass
class SubBlock:
    """サブブロック。親ブロックの一部."""

    start: datetime
    end: datetime
    messages: list[str]
    branches: list[str]
    turns: int
    keywords: list[str]  # このサブブロックの特徴キーワード
    is_transition: bool = False  # 遷移ゾーンか


@dataclass
class Block:
    """連続した作業ブロック."""

    start: datetime
    end: datetime
    projects: set[str]
    messages: list[str]
    branches: list[str]
    pr_numbers: list[int]
    turns: int


def build_blocks(
    events: list[Event],
    idle_threshold: int = 20,
    per_turn_time: int = 5,
) -> list[Block]:
    """イベント列を作業ブロックに変換."""
    if not events:
        return []

    blocks: list[Block] = []
    block_start = events[0].ts
    block_projs: set[str] = {events[0].project}
    block_msgs = [events[0].content]
    block_branches = [events[0].branch]
    block_prs = list(events[0].pr_numbers)
    prev_ts = events[0].ts
    turns = 1

    for ev in events[1:]:
        gap = (ev.ts - prev_ts).total_seconds() / 60
        if gap > idle_threshold:
            block_end = prev_ts + timedelta(minutes=per_turn_time)
            blocks.append(
                Block(
                    start=block_start,
                    end=block_end,
                    projects=block_projs.copy(),
                    messages=block_msgs.copy(),
                    branches=block_branches.copy(),
                    pr_numbers=block_prs.copy(),
                    turns=turns,
                )
            )
            block_start = ev.ts
            block_projs = set()
            block_msgs = []
            block_branches = []
            block_prs = []
            turns = 0
        block_projs.add(ev.project)
        block_msgs.append(ev.content)
        block_branches.append(ev.branch)
        block_prs.extend(ev.pr_numbers)
        turns += 1
        prev_ts = ev.ts

    block_end = prev_ts + timedelta(minutes=per_turn_time)
    blocks.append(
        Block(
            start=block_start,
            end=block_end,
            projects=block_projs.copy(),
            messages=block_msgs.copy(),
            branches=block_branches.copy(),
            pr_numbers=block_prs.copy(),
            turns=turns,
        )
    )
    return blocks


def calc_gross_time(
    events: list[Event],
) -> dict[str, tuple[datetime, datetime]]:
    """日別のグロス時間を算出."""
    daily: dict[str, list[datetime]] = defaultdict(list)
    for ev in events:
        date_key = ev.ts.strftime("%m/%d (%a)")
        daily[date_key].append(ev.ts)
    return {k: (min(v), max(v)) for k, v in daily.items()}


def format_duration(minutes: float) -> str:
    """分数を h:mm 形式に変換."""
    h = int(minutes // 60)
    m = int(minutes % 60)
    return f"{h}:{m:02d}"


def _cosine_sim_counters(a: Counter, b: Counter) -> float:
    """2つのCounterのコサイン類似度."""
    keys = set(a.keys()) | set(b.keys())
    if not keys:
        return 1.0
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def subdivide_blocks_by_keywords(
    blocks: list[Block],
    tfidf: dict[str, float],
    window_size: int = 5,
    similarity_threshold: float = 0.3,
    min_messages: int = 10,
) -> list[Block]:
    """ブロック内のメッセージ列に対してスライディングウィンドウの
    コサイン類似度を計算し、閾値以下なら分割する.

    Args:
        blocks: ブロックリスト
        tfidf: TF-IDFスコア辞書
        window_size: スライディングウィンドウサイズ
        similarity_threshold: 分割閾値（これ以下で分割）
        min_messages: これ未満のメッセージ数のブロックは分割しない
    """
    from ..analysis.tfidf import _tokenize

    result: list[Block] = []

    for block in blocks:
        if len(block.messages) < min_messages:
            result.append(block)
            continue

        # 各メッセージのTF-IDFベクトルをウィンドウ集計
        msg_tokens: list[list[str]] = []
        for msg in block.messages:
            msg_tokens.append(_tokenize(msg))

        # 分割点を検出
        split_points: list[int] = []
        n_msgs = len(block.messages)

        for i in range(window_size, n_msgs - window_size + 1):
            left_counter: Counter = Counter()
            for j in range(i - window_size, i):
                for token in msg_tokens[j]:
                    score = tfidf.get(token, 0.0)
                    if score > 0:
                        left_counter[token] += score

            right_counter: Counter = Counter()
            for j in range(i, min(i + window_size, n_msgs)):
                for token in msg_tokens[j]:
                    score = tfidf.get(token, 0.0)
                    if score > 0:
                        right_counter[token] += score

            sim = _cosine_sim_counters(left_counter, right_counter)
            if sim < similarity_threshold:
                # 前の分割点から十分離れているかチェック
                if not split_points or (i - split_points[-1]) >= window_size:
                    split_points.append(i)

        if not split_points:
            result.append(block)
            continue

        # 分割実行
        segments = [0] + split_points + [n_msgs]
        total_dur = (block.end - block.start).total_seconds()

        for si in range(len(segments) - 1):
            seg_start_idx = segments[si]
            seg_end_idx = segments[si + 1]
            seg_msgs = block.messages[seg_start_idx:seg_end_idx]
            seg_branches = block.branches[seg_start_idx:seg_end_idx]

            # 時間を按分
            ratio_start = seg_start_idx / n_msgs
            ratio_end = seg_end_idx / n_msgs
            t_start = block.start + timedelta(seconds=total_dur * ratio_start)
            t_end = block.start + timedelta(seconds=total_dur * ratio_end)

            result.append(
                Block(
                    start=t_start,
                    end=t_end,
                    projects=block.projects.copy(),
                    messages=seg_msgs,
                    branches=seg_branches,
                    pr_numbers=block.pr_numbers.copy(),
                    turns=len(seg_msgs),
                )
            )

    return result


def subdivide_block_soft(
    block: Block,
    tfidf: dict[str, float],
    window_size: int = 5,
    similarity_threshold: float = 0.3,
    overlap_margin: int = 2,
    min_messages: int = 10,
) -> list[SubBlock]:
    """ソフト境界でサブブロック生成。遷移ゾーンは重複可能.

    Args:
        block: 対象ブロック
        tfidf: TF-IDFスコア辞書
        window_size: スライディングウィンドウサイズ
        similarity_threshold: 分割閾値（これ以下で分割）
        overlap_margin: 遷移ゾーンの前後何メッセージを重複させるか
        min_messages: これ未満のメッセージ数は分割しない
    """
    from ..analysis.tfidf import _tokenize, extract_keywords

    n_msgs = len(block.messages)
    if n_msgs < min_messages:
        kws = extract_keywords(block.messages, top_n=5, tfidf_scores=tfidf)
        return [
            SubBlock(
                start=block.start,
                end=block.end,
                messages=block.messages[:],
                branches=block.branches[:],
                turns=block.turns,
                keywords=[w for w, _ in kws],
            )
        ]

    # 各メッセージのトークン列
    msg_tokens: list[list[str]] = [_tokenize(msg) for msg in block.messages]

    # 分割点を検出
    split_points: list[int] = []
    for i in range(window_size, n_msgs - window_size + 1):
        left_counter: Counter = Counter()
        for j in range(i - window_size, i):
            for token in msg_tokens[j]:
                score = tfidf.get(token, 0.0)
                if score > 0:
                    left_counter[token] += score

        right_counter: Counter = Counter()
        for j in range(i, min(i + window_size, n_msgs)):
            for token in msg_tokens[j]:
                score = tfidf.get(token, 0.0)
                if score > 0:
                    right_counter[token] += score

        sim = _cosine_sim_counters(left_counter, right_counter)
        if sim < similarity_threshold:
            if not split_points or (i - split_points[-1]) >= window_size:
                split_points.append(i)

    if not split_points:
        kws = extract_keywords(block.messages, top_n=5, tfidf_scores=tfidf)
        return [
            SubBlock(
                start=block.start,
                end=block.end,
                messages=block.messages[:],
                branches=block.branches[:],
                turns=block.turns,
                keywords=[w for w, _ in kws],
            )
        ]

    # サブブロック生成（遷移ゾーン付き）
    segments = [0] + split_points + [n_msgs]
    total_dur = (block.end - block.start).total_seconds()
    result: list[SubBlock] = []

    for si in range(len(segments) - 1):
        seg_start_idx = segments[si]
        seg_end_idx = segments[si + 1]

        # 時間を按分
        ratio_start = seg_start_idx / n_msgs
        ratio_end = seg_end_idx / n_msgs
        t_start = block.start + timedelta(seconds=total_dur * ratio_start)
        t_end = block.start + timedelta(seconds=total_dur * ratio_end)

        seg_msgs = block.messages[seg_start_idx:seg_end_idx]
        seg_branches = block.branches[seg_start_idx:seg_end_idx]
        kws = extract_keywords(seg_msgs, top_n=5, tfidf_scores=tfidf)

        result.append(
            SubBlock(
                start=t_start,
                end=t_end,
                messages=seg_msgs,
                branches=seg_branches,
                turns=len(seg_msgs),
                keywords=[w for w, _ in kws],
                is_transition=False,
            )
        )

        # 遷移ゾーンを挿入（最後のセグメント境界以外）
        if si < len(segments) - 2 and overlap_margin > 0:
            trans_start_idx = max(seg_end_idx - overlap_margin, seg_start_idx)
            trans_end_idx = min(seg_end_idx + overlap_margin, n_msgs)
            if trans_start_idx < trans_end_idx:
                ratio_ts = trans_start_idx / n_msgs
                ratio_te = trans_end_idx / n_msgs
                t_ts = block.start + timedelta(seconds=total_dur * ratio_ts)
                t_te = block.start + timedelta(seconds=total_dur * ratio_te)
                trans_msgs = block.messages[trans_start_idx:trans_end_idx]
                trans_branches = block.branches[trans_start_idx:trans_end_idx]
                trans_kws = extract_keywords(
                    trans_msgs, top_n=5, tfidf_scores=tfidf
                )
                result.append(
                    SubBlock(
                        start=t_ts,
                        end=t_te,
                        messages=trans_msgs,
                        branches=trans_branches,
                        turns=len(trans_msgs),
                        keywords=[w for w, _ in trans_kws],
                        is_transition=True,
                    )
                )

    return result
