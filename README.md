# ⏱️ claude-timecard

> Claude Codeのセッションログから、「いつ・何を・どのくらい」やったかを自動で出すCLI + MCPサーバー

vibe codingしてるフリーランスの**月末タイムシート作成を5分で終わらせる**ために作った。

---

## ✨ できること

| 機能 | 説明 |
|------|------|
| 📊 **日別Active時間** | メッセージ間隔からブロック構築 → キーワード細分化 → 正味時間算出 |
| 🌿 **ブランチ別集計** | どのブランチに何時間使ったか、重複なしで集計 |
| 🏷️ **キーワード抽出** | TF-IDF (unigram + bigram) で「何の作業か」を自動分類 |
| 🤖 **AIレポート** | Haiku(並列) → Opus(統合) で週報を自動生成。約120円/回 |
| 📈 **グラフ** | KDE密度曲線、キーワードランキング、タイムライン (PNG) |
| 🔌 **MCPサーバー** | Claude Codeから自然言語でクエリ（8ツール） |

---

## 🚀 30秒で始める

```bash
# 直近7日の作業時間
python timecard.py

# 特定プロジェクト、特定期間
python timecard.py -p myproject -s 2025-03-01 -e 2025-03-31

# ブランチ別 + ストリーム分析
python timecard.py -p myproject --by-branch --streams

# タスク推定 + グラフ4枚
python timecard.py -p myproject --tasks --graph

# AIレポート生成
python timecard.py -p myproject --tasks --report
```

出力は `output/{YYYYMMDD_HHMMSS}/` に整理される。

---

## 🔌 MCPサーバーとして使う

Claude Codeの中から「今月の作業時間は？」と聞くだけ。

### セットアップ

`~/.claude.json` の `mcpServers` に以下を追加:

```json
{
  "mcpServers": {
    "timecard": {
      "type": "stdio",
      "command": "python",
      "args": ["/path/to/mcp_server.py"]
    }
  }
}
```

> 💡 `/path/to/mcp_server.py` はこのリポジトリをクローンした場所に合わせて書き換えてください。
> 例: `"args": ["C:/Users/you/tools/claude-timecard/mcp_server.py"]`

### 使えるツール

| ツール | 聞き方の例 |
|--------|-----------|
| `timecard_daily` | 「今週の日別Active教えて」 |
| `timecard_detail` | 「3/23の午後に何やってた？」 |
| `timecard_streams` | 「ブランチ別の時間は？」 |
| `timecard_tasks` | 「今月のタスク一覧」 |
| `timecard_keywords` | 「よく使ってたキーワードは？」 |
| `timecard_compare_months` | 「先月と今月の比較」 |
| `timecard_report` | 「週報作って」 |
| `timecard_graph` | 「グラフ出して」 |

MCPサーバーが起動すると、Claude Codeがこれらのツールを自動認識する。あとは自然言語で聞くだけ。

---

## 🔧 カスタマイズ

### 同義語の登録

業務で使う専門用語を登録すると、キーワード抽出の精度が上がる。

```bash
# deploy と デプロイ を同じキーワードとして扱う
python timecard.py --add-synonym deploy デプロイ デプロイメント

# review と レビュー を統合
python timecard.py --add-synonym review レビュー コードレビュー
```

- 登録先: `~/.config/claude-timecard/synonyms.json`（ユーザーローカル）
- リポジトリの `synonyms.json` には汎用的な同義語のみ同梱
- **業務固有の用語はローカルに追加する**（公開リポジトリに含まれない）

> 💡 Claude Codeと対話しながら「この用語を同義語に追加して」と頼むこともできる。MCPサーバー経由でtimecard CLIを呼び出せば、会話の中で登録が完了する。

### Sudachi同義語辞書の一括インポート

[Sudachi同義語辞書](https://github.com/WorksApplications/SudachiDict)（25,000グループ、Apache-2.0）を一括インポートできる:

```bash
# URLから直接インポート
python timecard.py --import-sudachi https://raw.githubusercontent.com/WorksApplications/SudachiDict/develop/src/main/text/synonyms.txt

# ローカルファイルからインポート
python timecard.py --import-sudachi ~/Downloads/synonyms.txt
```

ユーザーローカル (`~/.config/claude-timecard/synonyms.json`) に保存される。既存の登録とマージされる。

### ストップワードの追加

キーワードに出てほしくない語を除外:

```bash
python timecard.py --add-stop 除外したい語1 除外したい語2
```

### ユーザー名の自動除外

OSのユーザー名（`$USERNAME` / `$USER`）は自動的にストップワードに追加される。キーワードにユーザー名が出ることはない。

---

## 📁 しくみ

```
~/.claude/projects/*.jsonl
    ↓
① JSONL解析         type:"user" のメッセージを構造化
    ↓
② idle分割          20分以上の空白でブロック分割
    ↓
③ キーワード細分化   同一ブロック内のトピック変化を検出してさらに分割
    ↓
④ プロジェクトフィルタ  特定クライアントの作業だけに絞る
    ↓
⑤ union-of-intervals  ブランチ間の重複時間を除去
    ↓
⑥ タスク推定         ブランチ名 + キーワード分布で自動分類
    ↓
⑦ AIレポート(opt)    Haiku(並列5) → Opus(統合)。SHA256キャッシュ付き
    ↓
出力: 日別レポート / JSON / グラフ / Markdown
```

---

## 🏗️ アーキテクチャ

```
timecard.py              # CLI（薄いエントリポイント）
mcp_server.py            # MCPサーバー（8ツール）
lib/
├── config.py            # 設定 + 定数のfactory
├── timing.py            # 処理時間計測
├── parser/              # JSONL解析（独立ライブラリとしても使える）
├── analysis/
│   ├── blocks.py        # ブロック構築 + キーワード細分化
│   ├── streams.py       # dir×branch グループ化
│   ├── intervals.py     # union-of-intervals (sweep line)
│   ├── tfidf.py         # TF-IDF（Registry経由で抽出器差替）
│   ├── kde.py           # カーネル密度推定（Gaussian/Epanechnikov）
│   ├── tasks.py         # ブランチベース タスク推定
│   └── extractors/      # 🔌 Registry: unigram, bigram, branch
├── signals/             # 🔌 Registry: branch, idle, keyword
├── calibration/         # 補正係数の学習（過去実績から）
├── report/              # Haiku/Opus レポート生成（並列+キャッシュ）
└── visualization/       # matplotlibグラフ（KDE, ランキング, タイムライン）
tests/                   # 55テスト
```

🔌マークはRegistryパターン。ファイル1つ追加 + `@register_xxx("名前")` で拡張可能。

---

## 🤖 AI向け情報（Claude Code / MCP）

このツールは Claude Code のセッションログ (`~/.claude/projects/{project-hash}/*.jsonl`) を解析します。

### データモデル

| 概念 | 説明 |
|------|------|
| **Event** | `type: "user"` のメッセージのみ抽出。タイムスタンプはUTC→JSTに変換 |
| **Block** | メッセージ間隔20分以内を1ブロック。最後のメッセージ+5分が終了時刻 |
| **SubBlock** | TF-IDFベクトルのコサイン類似度でブロック内をさらに細分化 |
| **Stream** | `(project_dir, branch)` のペアでグループ化。union-of-intervalsで重複除去 |
| **Active時間** | 全Streamのunionの合計分数。プロジェクトフィルタ適用後の正味値 |

### 同義語・ストップワードの管理

ユーザーの業務ドメインに合わせてキーワード精度を上げるために、以下のAPIがある:

```bash
# 同義語追加（ユーザーローカル: ~/.config/claude-timecard/synonyms.json）
python timecard.py --add-synonym <正規形> <別名1> [別名2 ...]

# ストップワード追加（stopwords.json の "ユーザー追加" カテゴリ）
python timecard.py --add-stop <語1> [語2 ...]
```

ユーザーとの対話の中で「このキーワードは別名として統合したい」「この語はノイズだから除外したい」という要望が出たら、上記コマンドで登録できる。登録後は次回の解析から反映される。

### MCP実装上の注意

MCPサーバー (`mcp_server.py`) は全ツール関数で `contextlib.redirect_stdout(sys.stderr)` を使用。`print()` がstdio JSON-RPCを壊すのを防止している。

---

## 📋 要件

- **Python 3.12+**
- **標準ライブラリのみ**（以下はオプション）
  - `matplotlib` → グラフ生成 (`--graph`)
  - `mcp` → MCPサーバー (`mcp_server.py`)
  - Claude CLI → AIレポート (`--report`)

## 📄 ライセンス

**AGPL-3.0** — [全文](./LICENSE)

| 利用形態 | 条件 |
|---------|------|
| 🙋 個人利用 | **無料**。AGPL-3.0の条件に従ってください |
| 🏢 企業・組織利用 | **商用ライセンス**が必要（[詳細](./COMMERCIAL_LICENSE.md)） |
| 🔧 派生物・改変 | ソースコードをAGPL-3.0で公開すれば無料で利用可能 |

企業でソースを非公開のまま利用したい場合は[商用ライセンス](./COMMERCIAL_LICENSE.md)をお問い合わせください。
