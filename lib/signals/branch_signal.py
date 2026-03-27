"""dir×branch切り替えシグナル（ハード境界）."""

from ..parser.events import Event
from ..config import TimecardConfig
from .base import Boundary
from .registry import register_signal


@register_signal("branch")
class BranchSignal:
    """ブランチ切り替えをハード境界として検出."""

    name = "branch"
    confidence = "hard"

    def detect(self, events: list[Event], config: TimecardConfig) -> list[Boundary]:
        """隣接イベント間でブランチまたはプロジェクトが変わった点を検出."""
        boundaries: list[Boundary] = []
        for i in range(1, len(events)):
            prev = events[i - 1]
            curr = events[i]
            if (prev.project, prev.branch) != (curr.project, curr.branch):
                if prev.branch and curr.branch:  # 両方ブランチ情報がある場合のみ
                    boundaries.append(
                        Boundary(
                            timestamp=curr.ts,
                            signal_name=self.name,
                            confidence=self.confidence,
                            reason=f"{prev.project}:{prev.branch} → {curr.project}:{curr.branch}",
                        )
                    )
        return boundaries
