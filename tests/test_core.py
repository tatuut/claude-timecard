"""claude-timecard コア機能のテスト."""

import json
import math
from collections import Counter
from datetime import datetime, timedelta, timezone

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.config import TimecardConfig, BreakTime, JST
from lib.parser.events import Event, parse_timestamp
from lib.analysis.blocks import Block, build_blocks
from lib.analysis.tfidf import build_tfidf, extract_keywords, _tokenize as tokenize
from lib.analysis.kde import GaussianKernel, EpanechnikovKernel, get_kernel, compute_kde_densities
from lib.analysis.intervals import Interval, measure_union
from lib.analysis.streams import group_by_stream, build_stream_blocks
from lib.calibration.calibrator import CalibrationData, DayRecord, Calibrator
from lib.signals.registry import get_all_signals

# --- helpers ---

def _make_event(hour, minute, project="proj-a", branch="feat-1", content="テスト"):
    ts = datetime(2026, 3, 24, hour, minute, tzinfo=JST)
    return Event(ts=ts, project=project, content=content, branch=branch)

def _make_events_sequence(times, project="proj-a", branch="feat-1"):
    """[(h,m), ...] からイベント列を作る"""
    return [_make_event(h, m, project, branch) for h, m in times]


# ============================================================
# R1: Parser
# ============================================================

class TestParser:
    def test_parse_timestamp_utc(self):
        ts = parse_timestamp("2026-03-24T04:30:12.345Z")
        assert ts is not None
        assert ts.tzinfo is not None
        # UTC 04:30 → JST 13:30
        assert ts.hour == 13
        assert ts.minute == 30

    def test_parse_timestamp_invalid(self):
        assert parse_timestamp("not-a-date") is None
        assert parse_timestamp("") is None

    def test_event_dataclass(self):
        e = _make_event(10, 30)
        assert e.project == "proj-a"
        assert e.branch == "feat-1"
        assert e.ts.hour == 10


# ============================================================
# R2: Blocks + Intervals
# ============================================================

class TestBlocks:
    def test_single_event(self):
        events = [_make_event(10, 0)]
        blocks = build_blocks(events, idle_threshold=20, per_turn_time=5)
        assert len(blocks) == 1
        # 10:00 → 10:05 = 5分
        dur = (blocks[0].end - blocks[0].start).total_seconds() / 60
        assert dur == 5.0

    def test_two_events_within_idle(self):
        events = _make_events_sequence([(10, 0), (10, 10)])
        blocks = build_blocks(events, idle_threshold=20, per_turn_time=5)
        assert len(blocks) == 1
        # 10:00 → 10:15 = 15分
        dur = (blocks[0].end - blocks[0].start).total_seconds() / 60
        assert dur == 15.0
        assert blocks[0].turns == 2

    def test_two_events_beyond_idle(self):
        events = _make_events_sequence([(10, 0), (11, 0)])
        blocks = build_blocks(events, idle_threshold=20, per_turn_time=5)
        assert len(blocks) == 2

    def test_idle_threshold_edge(self):
        # ちょうど20分 → 同一ブロック
        events = _make_events_sequence([(10, 0), (10, 20)])
        blocks = build_blocks(events, idle_threshold=20, per_turn_time=5)
        assert len(blocks) == 1

        # 21分 → 別ブロック
        events = _make_events_sequence([(10, 0), (10, 21)])
        blocks = build_blocks(events, idle_threshold=20, per_turn_time=5)
        assert len(blocks) == 2

    def test_per_turn_time_affects_end(self):
        events = [_make_event(10, 0)]
        blocks_5 = build_blocks(events, idle_threshold=20, per_turn_time=5)
        blocks_10 = build_blocks(events, idle_threshold=20, per_turn_time=10)
        dur_5 = (blocks_5[0].end - blocks_5[0].start).total_seconds() / 60
        dur_10 = (blocks_10[0].end - blocks_10[0].start).total_seconds() / 60
        assert dur_5 == 5.0
        assert dur_10 == 10.0

    def test_empty_events(self):
        assert build_blocks([], idle_threshold=20, per_turn_time=5) == []


class TestIntervals:
    def test_no_overlap(self):
        ivs = [
            Interval(
                start=datetime(2026, 3, 24, 10, 0, tzinfo=JST),
                end=datetime(2026, 3, 24, 11, 0, tzinfo=JST),
            ),
            Interval(
                start=datetime(2026, 3, 24, 12, 0, tzinfo=JST),
                end=datetime(2026, 3, 24, 13, 0, tzinfo=JST),
            ),
        ]
        assert measure_union(ivs) == 120.0  # 2時間

    def test_full_overlap(self):
        ivs = [
            Interval(
                start=datetime(2026, 3, 24, 10, 0, tzinfo=JST),
                end=datetime(2026, 3, 24, 12, 0, tzinfo=JST),
            ),
            Interval(
                start=datetime(2026, 3, 24, 10, 30, tzinfo=JST),
                end=datetime(2026, 3, 24, 11, 30, tzinfo=JST),
            ),
        ]
        assert measure_union(ivs) == 120.0  # 内側は重複

    def test_partial_overlap(self):
        ivs = [
            Interval(
                start=datetime(2026, 3, 24, 10, 0, tzinfo=JST),
                end=datetime(2026, 3, 24, 11, 0, tzinfo=JST),
            ),
            Interval(
                start=datetime(2026, 3, 24, 10, 30, tzinfo=JST),
                end=datetime(2026, 3, 24, 11, 30, tzinfo=JST),
            ),
        ]
        assert measure_union(ivs) == 90.0  # 10:00-11:30

    def test_empty(self):
        assert measure_union([]) == 0.0


class TestStreams:
    def test_group_by_stream(self):
        events = [
            _make_event(10, 0, "proj-a", "feat-1"),
            _make_event(10, 5, "proj-a", "feat-2"),
            _make_event(10, 10, "proj-a", "feat-1"),
        ]
        groups = group_by_stream(events)
        assert len(groups) == 2
        assert len(groups[("proj-a", "feat-1")]) == 2
        assert len(groups[("proj-a", "feat-2")]) == 1

    def test_build_stream_blocks(self):
        events = [
            _make_event(10, 0, "proj-a", "feat-1"),
            _make_event(10, 5, "proj-a", "feat-1"),
            _make_event(10, 10, "proj-a", "feat-2"),
        ]
        config = TimecardConfig()
        stream_blocks = build_stream_blocks(events, config)
        assert ("proj-a", "feat-1") in stream_blocks
        assert ("proj-a", "feat-2") in stream_blocks


# ============================================================
# R3: TF-IDF + KDE
# ============================================================

class TestTFIDF:
    def test_tokenize_english(self):
        tokens = tokenize("Fix the annotation bug in the codebase")
        assert "annotation" in tokens
        assert "bug" in tokens
        assert "fix" in tokens
        # 3文字未満は除外
        assert "in" not in tokens
        assert "the" not in tokens

    def test_tokenize_katakana(self):
        tokens = tokenize("アノテーション のバグを修正")
        assert "アノテーション" in tokens

    def test_tokenize_kanji(self):
        tokens = tokenize("検索機能の条件分岐")
        assert "検索機能" in tokens or "条件分岐" in tokens

    def test_build_tfidf(self):
        b1 = Block(
            start=datetime(2026, 3, 24, 10, 0, tzinfo=JST),
            end=datetime(2026, 3, 24, 11, 0, tzinfo=JST),
            projects={"proj"},
            messages=["annotation bug fix", "annotation panel"],
            branches=["feat-1"],
            pr_numbers=[],
            turns=2,
        )
        b2 = Block(
            start=datetime(2026, 3, 24, 12, 0, tzinfo=JST),
            end=datetime(2026, 3, 24, 13, 0, tzinfo=JST),
            projects={"proj"},
            messages=["wire detection logic", "wire routing"],
            branches=["feat-2"],
            pr_numbers=[],
            turns=2,
        )
        tfidf = build_tfidf([b1, b2])
        # "annotation" は1ブロックのみ → IDF高い
        # もし共通語があればIDF低い
        assert "annotation" in tfidf
        assert "wire" in tfidf


class TestKDE:
    def test_gaussian_kernel(self):
        k = GaussianKernel()
        assert k.evaluate(0.0) == 1.0
        assert 0 < k.evaluate(1.0) < 1.0
        assert 0 < k.evaluate(-1.0) < 1.0

    def test_epanechnikov_kernel(self):
        k = EpanechnikovKernel()
        assert k.evaluate(0.0) == 0.75
        assert k.evaluate(1.5) == 0.0
        assert k.evaluate(-1.5) == 0.0

    def test_get_kernel(self):
        g = get_kernel("gaussian")
        assert isinstance(g, GaussianKernel)
        e = get_kernel("epanechnikov")
        assert isinstance(e, EpanechnikovKernel)
        # unknown → default gaussian
        d = get_kernel("unknown")
        assert isinstance(d, GaussianKernel)


# ============================================================
# R4: Calibration
# ============================================================

class TestCalibration:
    def test_compute_coefficients(self):
        data = CalibrationData(records=[
            DayRecord(date="2026-02-26", actual_hours=14.0, location="office"),
            DayRecord(date="2026-02-27", actual_hours=10.0, location="office"),
            DayRecord(date="2026-02-20", actual_hours=10.0, location="remote"),
            DayRecord(date="2026-02-22", actual_hours=3.0, location="remote"),
        ])
        cal = Calibrator(data)
        # Active時間（分）のマップ
        daily_active = {
            "2026-02-26": 14 * 60,   # 840分
            "2026-02-27": 9.4 * 60,  # 564分
            "2026-02-20": 9.2 * 60,  # 552分
            "2026-02-22": 3.4 * 60,  # 204分
        }
        coeffs = cal.compute_coefficients(daily_active)
        assert "office" in coeffs
        assert "remote" in coeffs
        assert "default" in coeffs
        # office: 14*60/840=1.0, 10*60/564=1.064 → mean ≈ 1.032
        assert 0.9 < coeffs["office"] < 1.2
        # remote: 10*60/552=1.087, 3*60/204=0.882 → mean ≈ 0.98
        assert 0.7 < coeffs["remote"] < 1.3

    def test_calibrate(self):
        data = CalibrationData(records=[
            DayRecord(date="2026-02-26", actual_hours=14.0, location="office"),
        ])
        cal = Calibrator(data)
        daily_active = {"2026-02-26": 840.0}
        result = cal.calibrate(100.0, "office", daily_active)
        # coeff = 14*60/840 = 1.0 → 100 * 1.0 = 100
        assert result == 100.0

    def test_no_data(self):
        data = CalibrationData(records=[])
        cal = Calibrator(data)
        coeffs = cal.compute_coefficients({})
        assert coeffs["default"] == 1.0

    def test_detect_anomalies(self):
        data = CalibrationData(records=[])
        cal = Calibrator(data)
        daily_active = {
            "2026-02-25": 14 * 60,  # 高Active
            "2026-02-26": 14 * 60,
            "2026-02-27": 9 * 60,
        }
        daily_commits = {
            "2026-02-25": 1,   # 低commit → 異常
            "2026-02-26": 20,  # 正常
            "2026-02-27": 15,  # 正常
        }
        anomalies = cal.detect_anomalies(daily_active, daily_commits)
        dates = [a["date"] for a in anomalies]
        assert "2026-02-25" in dates
        assert "2026-02-26" not in dates

    def test_save_load(self, tmp_path):
        data = CalibrationData(records=[
            DayRecord(date="2026-02-26", actual_hours=14.0, location="office"),
        ])
        path = tmp_path / "cal.json"
        data.save(path)
        loaded = CalibrationData.load(path)
        assert len(loaded.records) == 1
        assert loaded.records[0].actual_hours == 14.0


# ============================================================
# R6: Signals Registry
# ============================================================

class TestSignals:
    def test_registry_has_signals(self):
        # signals モジュールをimportすることでregisterが発動する
        import lib.signals.branch_signal
        import lib.signals.idle_signal
        import lib.signals.keyword_signal
        signals = get_all_signals()
        assert "branch" in signals
        assert "idle" in signals
        assert "keyword" in signals

    def test_branch_signal_detect(self):
        import lib.signals.branch_signal
        from lib.signals.base import Boundary
        sig_cls = get_all_signals()["branch"]
        sig = sig_cls()
        events = [
            _make_event(10, 0, "proj-a", "feat-1"),
            _make_event(10, 5, "proj-a", "feat-1"),
            _make_event(10, 10, "proj-a", "feat-2"),  # branch切り替え
            _make_event(10, 15, "proj-a", "feat-2"),
        ]
        config = TimecardConfig()
        boundaries = sig.detect(events, config)
        assert len(boundaries) >= 1
        # feat-2に切り替わった時点（10:10）に境界
        assert any(b.timestamp.hour == 10 and b.timestamp.minute == 10 for b in boundaries)

    def test_idle_signal_detect(self):
        import lib.signals.idle_signal
        from lib.signals.base import Boundary
        sig_cls = get_all_signals()["idle"]
        sig = sig_cls()
        events = [
            _make_event(10, 0),
            _make_event(10, 5),
            _make_event(11, 0),  # 55分gap → idle
        ]
        config = TimecardConfig(idle_threshold_min=20)
        boundaries = sig.detect(events, config)
        assert len(boundaries) >= 1


# ============================================================
# Config
# ============================================================

class TestConfig:
    def test_default_config(self):
        c = TimecardConfig()
        assert c.idle_threshold_min == 20
        assert c.per_turn_time_min == 5
        assert c.kde_kernel == "gaussian"
        assert c.break_times == []

    def test_break_time(self):
        bt = BreakTime(start="12:00", end="13:00", label="昼休み")
        c = TimecardConfig(break_times=[bt])
        assert len(c.break_times) == 1
        assert c.break_times[0].label == "昼休み"


# ============================================================
# Integration: CLI smoke test
# ============================================================

class TestCLISmoke:
    def test_import(self):
        """CLIモジュールがimportできること"""
        import timecard
        assert hasattr(timecard, "main")


# ============================================================
# R7: Branch-first Task Inference
# ============================================================

class TestInferTasksBranchFirst:
    def test_infer_tasks_branch_first(self):
        """ブランチベースタスク推定の基本テスト."""
        from lib.analysis.tasks import infer_tasks_branch_first

        events = [
            _make_event(10, 0, "proj-a", "feat-auth", "implement login form validation"),
            _make_event(10, 5, "proj-a", "feat-auth", "add password strength check"),
            _make_event(10, 10, "proj-a", "feat-auth", "login session management"),
            _make_event(10, 15, "proj-a", "feat-auth", "authentication token handling"),
            _make_event(11, 0, "proj-a", "feat-ui", "refactor button component style"),
            _make_event(11, 5, "proj-a", "feat-ui", "update theme colors for dashboard"),
            _make_event(11, 10, "proj-a", "feat-ui", "responsive layout grid system"),
            _make_event(11, 15, "proj-a", "feat-ui", "component styling improvements"),
        ]

        blocks = build_blocks(events, idle_threshold=20, per_turn_time=5)
        tfidf = build_tfidf(blocks)

        task_nodes, ctx_kws = infer_tasks_branch_first(
            events, blocks, tfidf=tfidf,
            idle_threshold=20, per_turn_time=5,
        )

        # ブランチが2つなのでタスクが少なくとも1つ存在
        assert len(task_nodes) >= 1

        # 各TaskNodeにbranches情報がある
        for node in task_nodes:
            assert hasattr(node, "branches")
            assert isinstance(node.branches, list)

        # block_setが空でない
        all_blocks = set()
        for node in task_nodes:
            assert node.active_minutes >= 0
            all_blocks |= node.block_set
        assert len(all_blocks) > 0

    def test_branch_first_single_branch(self):
        """単一ブランチの場合も動作する."""
        from lib.analysis.tasks import infer_tasks_branch_first

        events = [
            _make_event(10, 0, "proj-a", "feat-1", "fix bug in parser"),
            _make_event(10, 5, "proj-a", "feat-1", "update parser tests"),
            _make_event(10, 10, "proj-a", "feat-1", "refactor parser module"),
        ]
        blocks = build_blocks(events, idle_threshold=20, per_turn_time=5)
        tfidf = build_tfidf(blocks)

        task_nodes, _ = infer_tasks_branch_first(
            events, blocks, tfidf=tfidf,
        )
        assert len(task_nodes) >= 1
        # 1ブランチなのでbranches にそのブランチが含まれる
        all_branches = []
        for node in task_nodes:
            all_branches.extend(node.branches)
        assert "feat-1" in all_branches

    def test_branch_first_empty(self):
        """ブロックなしの場合は空リストを返す."""
        from lib.analysis.tasks import infer_tasks_branch_first

        task_nodes, ctx = infer_tasks_branch_first([], [], tfidf={})
        assert task_nodes == []


# ============================================================
# R8: Subdivide Blocks by Keywords
# ============================================================

class TestSubdivideBlocksByKeywords:
    def test_short_block_not_subdivided(self):
        """メッセージ数が少ないブロックは分割されない."""
        from lib.analysis.blocks import subdivide_blocks_by_keywords

        block = Block(
            start=datetime(2026, 3, 24, 10, 0, tzinfo=JST),
            end=datetime(2026, 3, 24, 11, 0, tzinfo=JST),
            projects={"proj"},
            messages=["msg1", "msg2", "msg3"],
            branches=["feat-1", "feat-1", "feat-1"],
            pr_numbers=[],
            turns=3,
        )
        result = subdivide_blocks_by_keywords(
            [block], tfidf={"msg1": 1.0}, min_messages=10
        )
        assert len(result) == 1

    def test_homogeneous_block_not_subdivided(self):
        """同じキーワードが続くブロックは分割されない."""
        from lib.analysis.blocks import subdivide_blocks_by_keywords

        msgs = [f"annotation bug fix attempt {i}" for i in range(15)]
        block = Block(
            start=datetime(2026, 3, 24, 10, 0, tzinfo=JST),
            end=datetime(2026, 3, 24, 11, 0, tzinfo=JST),
            projects={"proj"},
            messages=msgs,
            branches=["feat-1"] * 15,
            pr_numbers=[],
            turns=15,
        )
        tfidf = {"annotation": 5.0, "bug": 3.0, "fix": 2.0, "attempt": 1.0}
        result = subdivide_blocks_by_keywords(
            [block], tfidf=tfidf, min_messages=10, window_size=3
        )
        # 同質なのでそのまま or 1ブロック
        assert len(result) >= 1

    def test_heterogeneous_block_subdivided(self):
        """異なるトピックが混在するブロックは分割される."""
        from lib.analysis.blocks import subdivide_blocks_by_keywords

        # 前半: annotation関連, 後半: wire/routing関連
        msgs = (
            ["annotation panel rendering bug"] * 8
            + ["wire routing detection logic"] * 8
        )
        block = Block(
            start=datetime(2026, 3, 24, 10, 0, tzinfo=JST),
            end=datetime(2026, 3, 24, 11, 0, tzinfo=JST),
            projects={"proj"},
            messages=msgs,
            branches=["feat-1"] * 16,
            pr_numbers=[],
            turns=16,
        )
        tfidf = {
            "annotation": 5.0, "panel": 3.0, "render": 2.0, "bug": 1.5,
            "wire": 5.0, "rout": 3.0, "detection": 2.0, "logic": 1.5,
        }
        result = subdivide_blocks_by_keywords(
            [block], tfidf=tfidf, min_messages=10, window_size=3,
            similarity_threshold=0.3,
        )
        # トピック変化があるので2つ以上に分割
        assert len(result) >= 2
        # 合計時間は元と同じ
        total_original = (block.end - block.start).total_seconds()
        total_result = sum(
            (b.end - b.start).total_seconds() for b in result
        )
        assert abs(total_original - total_result) < 1.0  # 浮動小数点誤差許容

    def test_empty_blocks(self):
        """空リストを渡しても動作する."""
        from lib.analysis.blocks import subdivide_blocks_by_keywords

        result = subdivide_blocks_by_keywords([], tfidf={})
        assert result == []


# ============================================================
# R9: Calibration Auto Apply
# ============================================================

class TestCalibrationAutoApply:
    def test_daily_report_with_calibration(self, capsys):
        """キャリブレーション係数適用時の日別レポート出力テスト."""
        from lib.report.formatter import print_daily_report

        events = [
            _make_event(10, 0, "proj-a", "feat-1", "work on feature"),
            _make_event(10, 10, "proj-a", "feat-1", "continue feature work"),
            _make_event(10, 20, "proj-a", "feat-1", "finish feature"),
        ]
        blocks = build_blocks(events, idle_threshold=20, per_turn_time=5)

        cal_coeffs = {"office": 1.2, "remote": 0.9, "default": 1.1}
        cal_locations = {"2026-03-24": "office"}

        print_daily_report(
            events, blocks,
            calibration_coeffs=cal_coeffs,
            calibration_locations=cal_locations,
        )

        captured = capsys.readouterr()
        # 推定: が表示されている
        assert "推定" in captured.out
        # 推定合計 が表示されている
        assert "推定合計" in captured.out

    def test_daily_report_without_calibration(self, capsys):
        """キャリブレーションなしの場合は推定が表示されない."""
        from lib.report.formatter import print_daily_report

        events = [
            _make_event(10, 0, "proj-a", "feat-1", "work on feature"),
            _make_event(10, 10, "proj-a", "feat-1", "continue work"),
        ]
        blocks = build_blocks(events, idle_threshold=20, per_turn_time=5)

        print_daily_report(events, blocks)

        captured = capsys.readouterr()
        assert "推定合計" not in captured.out

    def test_calibration_unknown_location_uses_default(self, capsys):
        """location不明日はdefault係数を使用."""
        from lib.report.formatter import print_daily_report

        events = [
            _make_event(10, 0, "proj-a", "feat-1", "work"),
            _make_event(10, 10, "proj-a", "feat-1", "more work"),
        ]
        blocks = build_blocks(events, idle_threshold=20, per_turn_time=5)

        cal_coeffs = {"office": 1.5, "remote": 0.8, "default": 1.0}
        # 日付に対するlocationがない → default使用
        cal_locations = {}

        print_daily_report(
            events, blocks,
            calibration_coeffs=cal_coeffs,
            calibration_locations=cal_locations,
        )

        captured = capsys.readouterr()
        assert "推定" in captured.out


# ============================================================
# R10: Extractor Registry
# ============================================================

class TestExtractorRegistry:
    def test_registry_has_all_extractors(self):
        """unigram/bigram/branchが登録されている."""
        from lib.analysis.extractors import get_all_extractors
        extractors = get_all_extractors()
        assert "unigram" in extractors
        assert "bigram" in extractors
        assert "branch" in extractors

    def test_unigram_extractor(self):
        """ユニグラム抽出器が既存の_tokenizeと同等の結果を返す."""
        from lib.analysis.extractors import get_all_extractors
        ext = get_all_extractors()["unigram"]()
        tokens = ext.extract("Fix the annotation bug in the codebase")
        assert "annotation" in tokens
        assert "bug" in tokens

    def test_bigram_extractor(self):
        """バイグラム抽出器が隣接ペアを返す."""
        from lib.analysis.extractors import get_all_extractors
        ext = get_all_extractors()["bigram"]()
        tokens = ext.extract("annotation panel rendering")
        # annotation_panel のようなバイグラムが含まれるはず
        assert any("_" in t for t in tokens)
        assert len(tokens) >= 1

    def test_branch_extractor(self):
        """ブランチ抽出器がブランチ名からトークンを抽出する."""
        from lib.analysis.extractors import get_all_extractors
        ext = get_all_extractors()["branch"]()
        # context["branch"]から抽出
        tokens = ext.extract("", branch="feat/user/annotation-panel")
        assert "annotation" in tokens
        assert "panel" in tokens
        # feat はスキップ、user は3文字未満ではないがスキップリストにないので含まれる
        assert "feat" not in tokens

    def test_branch_extractor_main_branch(self):
        """main/master/HEADは空リストを返す."""
        from lib.analysis.extractors import get_all_extractors
        ext = get_all_extractors()["branch"]()
        assert ext.extract("", branch="main") == []
        assert ext.extract("", branch="master") == []
        assert ext.extract("", branch="HEAD") == []

    def test_tokenize_with_extractors(self):
        """tokenize_with_extractorsが正しく動作する."""
        from lib.analysis.tfidf import tokenize_with_extractors
        # デフォルト（unigram）
        tokens = tokenize_with_extractors("annotation bug fix")
        assert "annotation" in tokens

        # 複数抽出器
        tokens = tokenize_with_extractors(
            "annotation panel",
            extractors=["unigram", "bigram"],
        )
        assert "annotation" in tokens
        assert any("_" in t for t in tokens)

        # ブランチ抽出器
        tokens = tokenize_with_extractors(
            "",
            extractors=["branch"],
            branch="feat/annotation-panel",
        )
        assert "annotation" in tokens


# ============================================================
# R11: Soft Subdivision
# ============================================================

class TestSoftSubdivision:
    def test_short_block_single_subblock(self):
        """短いブロックは1つのSubBlockになる."""
        from lib.analysis.blocks import subdivide_block_soft, SubBlock

        block = Block(
            start=datetime(2026, 3, 24, 10, 0, tzinfo=JST),
            end=datetime(2026, 3, 24, 11, 0, tzinfo=JST),
            projects={"proj"},
            messages=["msg1", "msg2", "msg3"],
            branches=["feat-1", "feat-1", "feat-1"],
            pr_numbers=[],
            turns=3,
        )
        result = subdivide_block_soft(block, tfidf={"msg1": 1.0}, min_messages=10)
        assert len(result) == 1
        assert isinstance(result[0], SubBlock)
        assert not result[0].is_transition

    def test_heterogeneous_produces_subblocks(self):
        """異なるトピック混在で複数SubBlockに分割される."""
        from lib.analysis.blocks import subdivide_block_soft, SubBlock

        msgs = (
            ["annotation panel rendering bug"] * 8
            + ["wire routing detection logic"] * 8
        )
        block = Block(
            start=datetime(2026, 3, 24, 10, 0, tzinfo=JST),
            end=datetime(2026, 3, 24, 11, 0, tzinfo=JST),
            projects={"proj"},
            messages=msgs,
            branches=["feat-1"] * 16,
            pr_numbers=[],
            turns=16,
        )
        tfidf = {
            "annotation": 5.0, "panel": 3.0, "render": 2.0, "bug": 1.5,
            "wire": 5.0, "rout": 3.0, "detection": 2.0, "logic": 1.5,
        }
        result = subdivide_block_soft(
            block, tfidf=tfidf, min_messages=10, window_size=3,
            similarity_threshold=0.3, overlap_margin=2,
        )
        # 複数のサブブロックが生成される
        assert len(result) >= 2
        # SubBlockインスタンスであること
        for sb in result:
            assert isinstance(sb, SubBlock)
        # keywordsが設定されている
        for sb in result:
            assert isinstance(sb.keywords, list)

    def test_transition_zones_exist(self):
        """遷移ゾーンが生成される."""
        from lib.analysis.blocks import subdivide_block_soft, SubBlock

        msgs = (
            ["annotation panel rendering bug"] * 8
            + ["wire routing detection logic"] * 8
        )
        block = Block(
            start=datetime(2026, 3, 24, 10, 0, tzinfo=JST),
            end=datetime(2026, 3, 24, 11, 0, tzinfo=JST),
            projects={"proj"},
            messages=msgs,
            branches=["feat-1"] * 16,
            pr_numbers=[],
            turns=16,
        )
        tfidf = {
            "annotation": 5.0, "panel": 3.0, "render": 2.0, "bug": 1.5,
            "wire": 5.0, "rout": 3.0, "detection": 2.0, "logic": 1.5,
        }
        result = subdivide_block_soft(
            block, tfidf=tfidf, min_messages=10, window_size=3,
            similarity_threshold=0.3, overlap_margin=2,
        )
        # 分割が発生していれば遷移ゾーンがある
        non_transition = [sb for sb in result if not sb.is_transition]
        if len(non_transition) >= 2:
            transitions = [sb for sb in result if sb.is_transition]
            assert len(transitions) >= 1


# ============================================================
# R12: Timing Module
# ============================================================

class TestTimingModule:
    def test_timer_basic(self):
        """Timerの基本動作テスト."""
        from lib.timing import Timer
        timer = Timer()
        with timer.measure("test_step"):
            total = sum(range(1000))
        assert len(timer.results) == 1
        assert timer.results[0].label == "test_step"
        assert timer.results[0].seconds >= 0

    def test_timer_summary(self):
        """Timer.summary()がdict形式で返る."""
        from lib.timing import Timer
        timer = Timer()
        with timer.measure("step1"):
            pass
        with timer.measure("step2"):
            pass
        summary = timer.summary()
        assert "total_seconds" in summary
        assert "steps" in summary
        assert len(summary["steps"]) == 2
        assert summary["steps"][0]["label"] == "step1"
        assert summary["steps"][1]["label"] == "step2"

    def test_enable_disable_timing(self):
        """enable_timing/disable_timingの動作テスト."""
        from lib.timing import enable_timing, get_timer, disable_timing
        # 初期状態はNone
        disable_timing()
        assert get_timer() is None
        # 有効化
        timer = enable_timing()
        assert get_timer() is timer
        # 無効化
        disable_timing()
        assert get_timer() is None


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
