# claude-timecard

Automatically extract work hours, tasks, and keywords from [Claude Code](https://claude.ai/claude-code) session logs.

Built for freelancers who bill by the hour and hate filling out timesheets.

## What it does

Claude Code stores every conversation as timestamped JSONL in `~/.claude/projects/`. This tool reads those logs and turns them into structured time reports — no manual tracking required.

```
~/.claude/projects/*.jsonl
    → JSONL parse → idle split (20min) → keyword subdivision (TF-IDF cosine)
    → project filter → union-of-intervals → task estimation
    → output: daily report / JSON / graphs / AI-generated weekly report
```

## Quick start

```bash
# Last 7 days, all projects
python timecard.py

# Filter by project, specific date range
python timecard.py -p myproject -s 2025-03-01 -e 2025-03-31

# Branch breakdown + stream analysis
python timecard.py -p myproject --by-branch --streams

# Task estimation + 4 PNG graphs
python timecard.py -p myproject --tasks --graph

# AI-generated report (Haiku parallel → Opus synthesis, ~$0.80/run)
python timecard.py -p myproject --tasks --report

# JSON output (pipe to jq, scripts, etc.)
python timecard.py --json
```

Output goes to `output/{YYYYMMDD_HHMMSS}/`.

## MCP server

Works as an MCP server for Claude Code. Add to `~/.claude.json`:

```json
{
  "mcpServers": {
    "timecard": {
      "type": "stdio",
      "command": "python",
      "args": ["/absolute/path/to/mcp_server.py"]
    }
  }
}
```

Then just ask Claude Code in natural language:

| Tool | Example |
|------|---------|
| `timecard_daily` | "Show me daily active hours this week" |
| `timecard_detail` | "What was I doing on 3/23 afternoon?" |
| `timecard_streams` | "Time per branch?" |
| `timecard_tasks` | "Task list for this month" |
| `timecard_keywords` | "Top keywords?" |
| `timecard_compare_months` | "Compare Feb vs March" |
| `timecard_report` | "Write a weekly report" |
| `timecard_graph` | "Generate graphs" |

## Customization

### Synonyms

Register domain-specific synonyms to improve keyword extraction:

```bash
python timecard.py --add-synonym deploy デプロイ デプロイメント
python timecard.py --add-synonym review レビュー コードレビュー
```

Stored in `~/.config/claude-timecard/synonyms.json` (user-local, never pushed to git).

### Bulk import from Sudachi

Import [Sudachi synonym dictionary](https://github.com/WorksApplications/SudachiDict) (25k groups, Apache-2.0):

```bash
python timecard.py --import-sudachi https://raw.githubusercontent.com/WorksApplications/SudachiDict/develop/src/main/text/synonyms.txt
```

### Stopwords

```bash
python timecard.py --add-stop word1 word2
```

OS username (`$USERNAME` / `$USER`) is automatically excluded from keywords.

## How it works

| Concept | Description |
|---------|-------------|
| **Event** | User messages extracted from JSONL. Timestamps converted UTC → JST |
| **Block** | Messages within 20min gaps grouped together. Last message + 5min = block end |
| **SubBlock** | Blocks further split where TF-IDF cosine similarity detects topic changes |
| **Stream** | `(project_dir, branch)` pairs. Union-of-intervals removes overlaps |
| **Active time** | Total minutes from the union of all streams after project filtering |

### Pipeline

```
JSONL → parse user messages → idle split (20min threshold)
     → keyword subdivision (TF-IDF cosine similarity)
     → project filter → union-of-intervals (sweep line)
     → task estimation (branch + keyword clustering)
     → optional: AI report (Haiku ×5 parallel → Opus synthesis, SHA256 cached)
```

## Architecture

```
timecard.py            CLI entry point
mcp_server.py          MCP server (8 tools, stdio transport)
lib/
├── config.py          Constants, TimecardConfig dataclass
├── timing.py          Performance measurement
├── parser/            JSONL parsing (standalone-capable)
├── analysis/
│   ├── blocks.py      Block construction + keyword subdivision
│   ├── streams.py     dir×branch stream grouping
│   ├── intervals.py   Union-of-intervals (sweep line)
│   ├── tfidf.py       TF-IDF + tokenization + synonym management
│   ├── kde.py         Kernel density estimation
│   ├── tasks.py       Branch-based task estimation
│   └── extractors/    Registry: unigram, bigram, branch
├── signals/           Registry: branch, idle, keyword
├── calibration/       Coefficient learning from past records
├── report/            Haiku/Opus report generation (parallel + cached)
└── visualization/     matplotlib graphs (KDE, ranking, timeline)
tests/                 55 tests
```

Registries (`extractors/`, `signals/`) are extensible — drop in a file + `@register("name")`.

## Requirements

- Python 3.12+
- **No required dependencies** (stdlib only)
- Optional: `matplotlib` (graphs), `mcp` (MCP server), Claude CLI (AI reports)

## For AI agents (Claude Code / MCP)

The MCP server wraps all tool functions with `contextlib.redirect_stdout(sys.stderr)` to prevent `print()` from corrupting the stdio JSON-RPC transport.

Synonym and stopword management commands:

```bash
# Add synonym (saved to ~/.config/claude-timecard/synonyms.json)
python timecard.py --add-synonym <canonical> <alias1> [alias2 ...]

# Add stopword (saved to stopwords.json under "ユーザー追加" category)
python timecard.py --add-stop <word1> [word2 ...]

# Import Sudachi synonym dictionary
python timecard.py --import-sudachi <path-or-url>
```

When a user asks to merge keywords or exclude noise words, use these commands. Changes take effect on the next analysis run.

## License

**AGPL-3.0** ([full text](./LICENSE))

- **Individual use**: Free under AGPL-3.0
- **Corporate use**: Commercial license required ([details](./COMMERCIAL_LICENSE.md))
- **Derivatives**: Must be published under AGPL-3.0 with full source

To use without source disclosure obligations, contact for a commercial license.
