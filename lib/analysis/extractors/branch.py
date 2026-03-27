"""ブランチ名抽出器: ブランチ名からトークンを抽出."""

import re

from .registry import register_extractor


@register_extractor("branch")
class BranchExtractor:
    """ブランチ名からトークンを抽出。context["branch"]を使用."""

    name = "branch"

    _SKIP = {"feat", "fix", "refactor", "feature", "chore", "hotfix"}

    def extract(self, text: str, **context) -> list[str]:
        branch = context.get("branch", "")
        if not branch or branch in ("main", "master", "HEAD", "development"):
            return []
        # ハイフン、スラッシュ、アンダースコアで分割
        parts = re.split(r'[/\-_]', branch)
        # 3文字以上、ユーザー名・prefix除外
        return [p.lower() for p in parts if len(p) >= 3 and p.lower() not in self._SKIP]
