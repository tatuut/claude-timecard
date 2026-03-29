---
name: timecard-assist
description: Claude Code MCPのtimecardツールを使って作業時間分析・タイムシート作成を支援。「今月の作業時間」「タイムシート」「月末処理」「何時間働いた？」「作業内容まとめ」に反応。
---

# timecard-assist — 作業時間分析・タイムシート支援

claude-timecard MCPサーバーのツールを使って、ユーザーのタイムシート作成を支援する。

## 使えるツール（3つだけ）

| ツール | 用途 |
|--------|------|
| `timecard_day` | 特定日の全部入りビュー（ブランチ別、キーワード、PR+コメント、コミット+ファイル変更、AI要約） |
| `timecard_conversation` | 指定日・時間帯の会話内容を取得（作業内容の根拠確認用） |
| `timecard_register` | 同義語/ストップワード/Sudachi辞書の登録 |

## ワークフロー

### Step 1: ヒアリング（必ず最初にやる）

推測で進めず、以下を対話で確認する:

1. **プロジェクト名** — 「どのプロジェクトの時間を見ますか？」
2. **対象期間** — 「今月分（3/1〜3/31）ですか？先月分ですか？」
3. **過去の手動記録** — 「過去に手動で記録したタイムシートはありますか？あれば形式を揃えられます」
4. **出力フォーマット** — 過去データがあればそれに合わせる。なければ聞く
5. **勤務状況** — 「体調不良で休んだ日は？」「休憩時間は？」「リモート/出社の週は？」

### Step 2: 日ごとにデータ取得

`timecard_day` を対象期間の各日で呼ぶ。返ってくるもの:
- ブランチ別グループ（時間、キーワード、PR情報、コミット情報）
- AI要約（`generate_excerpts=true` でHaikuが自動生成）

### Step 3: ノイズ検出

結果のキーワードを見て、ノイズがあれば `timecard_register` で対処:
- `ultrathink` → ストップワード追加
- 表記揺れ → 同義語登録

### Step 4: タイムシート作成

**AI要約 (`ai_excerpt`) を根拠に**作業内容を書く。ブランチ名やキーワードから推測しない。

要約が不十分な場合は `timecard_conversation` で実際の会話内容を確認する。

### Step 5: 確認・調整

生成したタイムシートをユーザーに見せて確認を取る。

## timecard_day のレスポンス構造

```json
{
  "date": "2026-03-27",
  "active_hours": 10.4,
  "branch_groups": [
    {
      "branch": "feat/tatut/dijkstra-improvement-v2",
      "projects": ["prefab-electrical-takeoff-app"],
      "active_hours": 5.1,
      "time_ranges": ["01:14~06:40"],
      "keywords": ["優先", "registry", "cost"],
      "pr_numbers": [234],
      "pr_info": [{"number": 234, "title": "...", "comments": [...]}],
      "commits": [{"hash": "abc1234", "message": "...", "files": [...]}],
      "ai_excerpt": "dijkstraの優先度コスト計算を改修..."
    }
  ]
}
```

## 注意事項

- Active時間はメッセージ間隔から推定。実際の請求時間とはズレがある
- 過去の手動記録があれば補正係数の計算に使える
- AI要約にはユーザーの業務情報が含まれる。正確性は必ずユーザーに確認する
