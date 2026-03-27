"""dir×branchグループ化, ストリーム分類."""

from collections import defaultdict

from ..parser.events import Event
from ..config import TimecardConfig
from .blocks import Block, build_blocks


def group_by_stream(events: list[Event]) -> dict[tuple[str, str], list[Event]]:
    """(dir, branch) ペアでイベントをグループ化."""
    groups: dict[tuple[str, str], list[Event]] = defaultdict(list)
    for e in events:
        key = (e.project, e.branch or "__no_branch__")
        groups[key].append(e)
    return dict(groups)


def build_stream_blocks(
    events: list[Event], config: TimecardConfig
) -> dict[tuple[str, str], list[Block]]:
    """各ストリームごとにブロック構築."""
    streams = group_by_stream(events)
    return {
        key: build_blocks(evts, config.idle_threshold_min, config.per_turn_time_min)
        for key, evts in streams.items()
    }
