import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import motivation.summarize_dynaextent_mixed_ab as analyzer
from motivation.test_summarize_live_extent_mixed_ab import make_report


DYNAMIC_LOGICAL_READS = 1000
DYNAMIC_FALLBACKS = 100
DYNAMIC_EXTRA_FULL_RETRIES = 25
DYNAMIC_PROMOTIONS = 80
DYNAMIC_DEMOTIONS = 50
SHORT_RECORD_BYTES = 420
FULL_RECORD_BYTES = 832


def _set_dynamic_telemetry(
        report, *, short_reads, full_reads, read_bytes,
        fallback_reads=0, promotions=0, demotions=0):
    gpu = report["gpu_persistent"]
    attempts = short_reads + full_reads
    values = {
        "dynamic_graph_short_reads": short_reads,
        "dynamic_graph_full_reads": full_reads,
        "dynamic_graph_read_bytes": read_bytes,
        "dynamic_graph_fallback_reads": fallback_reads,
        "dynamic_graph_hint_promotions": promotions,
        "dynamic_graph_hint_demotions": demotions,
        "dynamic_graph_snapshot_attempts": attempts,
        "dynamic_graph_nonfallback_full_attempts":
            max(full_reads - fallback_reads, 0),
        "dynamic_graph_short_physical_ratio":
            short_reads / attempts if attempts else 0.0,
        "dynamic_graph_fallback_ratio":
            fallback_reads / short_reads if short_reads else 0.0,
        "average_dynamic_graph_read_bytes_per_physical_read":
            read_bytes / attempts if attempts else 0.0,
        "average_dynamic_graph_read_bytes_per_query":
            read_bytes / gpu["queries_completed"],
    }
    gpu.update(values)


def _recompute_global_graph_averages(report):
    gpu = report["gpu_persistent"]
    gpu["average_graph_read_bytes_per_query"] = (
        gpu["graph_read_bytes"] / gpu["queries_completed"])
    gpu["average_graph_read_bytes_per_logical_parent"] = (
        gpu["graph_read_bytes"] / gpu["graph_page_requests"])


def make_triplet_report(mode):
    if mode not in analyzer.MODE_SPECS:
        raise ValueError(mode)
    spec = analyzer.MODE_SPECS[mode]
    report = make_report(spec.graph_policy)
    report["meta"]["gpu_dynamic_graph_extent"] = spec.dynamic_enabled
    report["meta"]["gpu_dynamic_graph_extent_source"] = spec.dynamic_source
    report["meta"]["benchmark_driver_concurrency"][
        "client_threads_source"] = "explicit"
    gpu = report["gpu_persistent"]
    # The legacy fixture predates these aggregate counters; current reports
    # always emit them and the triplet analyzer intentionally requires them.
    gpu.setdefault("graph_extent_underhint_reads", 0)
    gpu.setdefault("graph_extent_hint_promotions", 0)

    if mode == "fixed":
        _set_dynamic_telemetry(
            report,
            short_reads=0,
            full_reads=DYNAMIC_LOGICAL_READS,
            read_bytes=DYNAMIC_LOGICAL_READS * FULL_RECORD_BYTES,
        )
    elif mode == "static-only":
        gpu["graph_live_extent_reads"] -= DYNAMIC_LOGICAL_READS
        gpu["graph_full_record_reads"] += DYNAMIC_LOGICAL_READS
        byte_delta = DYNAMIC_LOGICAL_READS * (
            FULL_RECORD_BYTES - SHORT_RECORD_BYTES)
        gpu["graph_read_bytes"] += byte_delta
        gpu["rdma_read_bytes"] += byte_delta
        _recompute_global_graph_averages(report)
        _set_dynamic_telemetry(
            report,
            short_reads=0,
            full_reads=DYNAMIC_LOGICAL_READS,
            read_bytes=DYNAMIC_LOGICAL_READS * FULL_RECORD_BYTES,
        )
    else:
        extra_full = DYNAMIC_FALLBACKS + DYNAMIC_EXTRA_FULL_RETRIES
        gpu["graph_full_record_reads"] += extra_full
        gpu["graph_read_retries"] += extra_full
        gpu["graph_extent_fallback_reads"] = DYNAMIC_FALLBACKS
        gpu["graph_extent_underhint_reads"] = DYNAMIC_FALLBACKS
        gpu["graph_extent_hint_promotions"] = DYNAMIC_PROMOTIONS
        gpu["graph_extent_fallback_ratio"] = (
            DYNAMIC_FALLBACKS / gpu["graph_live_extent_reads"])
        gpu["graph_read_bytes"] += extra_full * FULL_RECORD_BYTES
        gpu["rdma_read_bytes"] += extra_full * FULL_RECORD_BYTES
        gpu["rdma_read_ops"] += extra_full
        _recompute_global_graph_averages(report)
        _set_dynamic_telemetry(
            report,
            short_reads=DYNAMIC_LOGICAL_READS,
            full_reads=extra_full,
            read_bytes=(
                DYNAMIC_LOGICAL_READS * SHORT_RECORD_BYTES +
                extra_full * FULL_RECORD_BYTES),
            fallback_reads=DYNAMIC_FALLBACKS,
            promotions=DYNAMIC_PROMOTIONS,
            demotions=DYNAMIC_DEMOTIONS,
        )
    return report


class DynaExtentMixedTripletAnalyzerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def write_report(
            self, mode, repeat=1, *, document=None, suffix="",
            snapshot_id=None):
        case_root = (
            self.root / mode / "concurrency_336" /
            f"repeat_{repeat}")
        directory = case_root / ("04_gpu_persistent_gpunetio" + suffix)
        directory.mkdir(parents=True, exist_ok=True)
        reset_log = case_root / analyzer.RESET_LOG_NAME
        latin_position = analyzer._expected_latin_position(mode, repeat)
        latin_cycle = (repeat - 1) // analyzer.LATIN_SQUARE_SIZE + 1
        snapshot_id = snapshot_id or f"snapshot-v1-repeat-{repeat:04d}"
        reset_log.write_text(
            "\n".join((
                "hook=/reset",
                f"policy={mode}",
                "concurrency=336",
                f"repetition={repeat}",
                f"latin_position={latin_position}",
                f"latin_cycle={latin_cycle}",
                f"snapshot_id={snapshot_id}",
                "exit_status=0",
                "",
            )),
            encoding="utf-8",
        )
        path = directory / f"sift100m_{mode}_{repeat}{suffix}.json"
        report = (
            make_triplet_report(mode) if document is None else document)
        report[analyzer.RESET_CERTIFICATE_KEY] = {
            "schema_version": 1,
            "snapshot_id": snapshot_id,
            "reset_log_sha256": hashlib.sha256(
                reset_log.read_bytes()).hexdigest(),
            "policy": mode,
            "concurrency": 336,
            "repetition": repeat,
            "latin_position": latin_position,
            "latin_cycle": latin_cycle,
        }
        path.write_text(json.dumps(report), encoding="utf-8")
        return path

    def write_triplet(self, repeat=1):
        for mode in analyzer.MODES:
            self.write_report(mode, repeat)

    def write_latin_cycles(self, cycles=1):
        for repeat in range(1, cycles * analyzer.LATIN_SQUARE_SIZE + 1):
            self.write_triplet(repeat)

    def test_runner_fails_closed_without_nonempty_reset_hook(self):
        runner = Path(__file__).with_name("run_dynaextent_mixed_ab.sh")
        environment = os.environ.copy()
        environment["REPETITIONS"] = "3"
        environment.pop("DYNAEXTENT_BEFORE_CASE_HOOK", None)
        missing = subprocess.run(
            ["bash", str(runner)],
            cwd=runner.parent.parent,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("is required", missing.stderr)

        empty_hook = self.root / "empty-reset-hook"
        empty_hook.touch(mode=0o755)
        environment["DYNAEXTENT_BEFORE_CASE_HOOK"] = str(empty_hook)
        empty = subprocess.run(
            ["bash", str(runner)],
            cwd=runner.parent.parent,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(empty.returncode, 0)
        self.assertIn("nonempty executable", empty.stderr)

        environment["REPETITIONS"] = "1"
        unbalanced = subprocess.run(
            ["bash", str(runner)],
            cwd=runner.parent.parent,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(unbalanced.returncode, 0)
        self.assertIn("multiple of 3", unbalanced.stderr)

    def test_runner_rejects_successful_hook_without_snapshot_id(self):
        runner = Path(__file__).with_name("run_dynaextent_mixed_ab.sh")
        hook = self.root / "reset-without-snapshot-id"
        hook.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        hook.chmod(0o755)
        index_prefix = self.root / "index"
        Path(str(index_prefix) + ".gextent8").write_bytes(b"sidecar")
        environment = os.environ.copy()
        environment.update({
            "REPETITIONS": "3",
            "DYNAEXTENT_BEFORE_CASE_HOOK": str(hook),
            "DYNAEXTENT_RESULT_ROOT": str(self.root / "results"),
            "PQ_INDEX_PREFIX": str(index_prefix),
        })
        completed = subprocess.run(
            ["bash", str(runner)],
            cwd=runner.parent.parent,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("must emit exactly one", completed.stderr)

    def test_valid_triplet_emits_all_three_comparisons_and_telemetry(self):
        self.write_latin_cycles()
        triplets = analyzer.discover_triplets(self.root)
        self.assertEqual(len(triplets), 3)
        summary = analyzer.build_summary(self.root, triplets)
        self.assertEqual(summary["triplet_count"], 3)
        self.assertEqual(
            summary["experiment_contract"]["complete_latin_cycles"], 1)
        self.assertEqual(
            set(summary["paired_medians"]),
            {"static_over_fixed", "dyna_over_static", "dyna_over_fixed"},
        )
        self.assertEqual(
            summary["policy_medians"]["dynaextent"]
            ["dynamic_graph_short_reads"],
            DYNAMIC_LOGICAL_READS,
        )
        query_count = make_triplet_report("dynaextent")["gpu_persistent"][
            "queries_completed"]
        self.assertAlmostEqual(
            summary["policy_medians"]["dynaextent"]
            ["dynamic_graph_short_reads_per_query"],
            DYNAMIC_LOGICAL_READS / query_count,
        )
        self.assertAlmostEqual(
            summary["policy_medians"]["dynaextent"]
            ["dynamic_graph_fallback_ratio"],
            DYNAMIC_FALLBACKS / DYNAMIC_LOGICAL_READS,
        )
        self.assertFalse(
            summary["dynamic_telemetry_semantics"]
            ["snapshot_attempts_are_logical_reads"])
        self.assertFalse(
            summary["dynamic_telemetry_semantics"]
            ["headline_uses_raw_totals"])
        for field in analyzer.DYNA_RAW_FIELDS:
            self.assertNotIn(field, analyzer.HEADLINE_METRICS)
        for field in analyzer.DYNA_PER_QUERY_FIELDS:
            self.assertIn(field, analyzer.HEADLINE_METRICS)
        markdown = analyzer.render_markdown(summary)
        self.assertIn("static/fixed", markdown)
        self.assertIn("Dyna/static", markdown)
        self.assertIn("physical snapshot attempts", markdown)
        self.assertIn("target attainment", markdown)

        json_path = self.root / "out.json"
        markdown_path = self.root / "out.md"
        self.assertEqual(analyzer.main([
            str(self.root),
            "--json-output", str(json_path),
            "--markdown-output", str(markdown_path),
        ]), 0)
        self.assertTrue(json_path.is_file())
        self.assertTrue(markdown_path.is_file())

    def test_rejects_incomplete_latin_cycle(self):
        self.write_triplet()
        with self.assertRaisesRegex(
                analyzer.ReportError, "whole 3x3 Latin-square cycle"):
            analyzer.discover_triplets(self.root)

    def test_rejects_snapshot_mismatch_and_tampered_reset_log(self):
        self.write_latin_cycles()
        dyna_report = (
            self.root / "dynaextent" / "concurrency_336" / "repeat_1" /
            "04_gpu_persistent_gpunetio" /
            "sift100m_dynaextent_1.json")
        dyna_report.unlink()
        self.write_report(
            "dynaextent", repeat=1,
            snapshot_id="different-snapshot-repeat-0001")
        with self.assertRaisesRegex(analyzer.ReportError, "snapshot_id mismatch"):
            analyzer.discover_triplets(self.root)

        self.temporary.cleanup()
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.write_latin_cycles()
        reset_log = (
            self.root / "fixed" / "concurrency_336" / "repeat_1" /
            analyzer.RESET_LOG_NAME)
        with reset_log.open("a", encoding="utf-8") as stream:
            stream.write("tampered=true\n")
        with self.assertRaisesRegex(analyzer.ReportError, "digest mismatch"):
            analyzer.discover_triplets(self.root)

    def test_rejects_report_bound_to_wrong_latin_position(self):
        self.write_latin_cycles()
        report_path = (
            self.root / "static-only" / "concurrency_336" / "repeat_2" /
            "04_gpu_persistent_gpunetio" /
            "sift100m_static-only_2.json")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report[analyzer.RESET_CERTIFICATE_KEY]["latin_position"] = 2
        report_path.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(
                analyzer.ReportError, "latin_position=2, expected 1"):
            analyzer.discover_triplets(self.root)

    def test_rejects_missing_policy_and_duplicate_report(self):
        self.write_report("fixed")
        self.write_report("static-only")
        with self.assertRaisesRegex(analyzer.ReportError, "missing policy"):
            analyzer.discover_triplets(self.root)

        self.temporary.cleanup()
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.write_triplet()
        self.write_report("fixed", suffix="_duplicate")
        with self.assertRaisesRegex(analyzer.ReportError, "duplicate reports"):
            analyzer.discover_triplets(self.root)

    def test_rejects_wrong_directory_to_report_mapping_and_non_bool_flag(self):
        for field, value, message in (
                ("gpu_query_graph_read_policy", "fixed",
                 "gpu_query_graph_read_policy"),
                ("gpu_dynamic_graph_extent", 1,
                 "gpu_dynamic_graph_extent"),
                ("gpu_dynamic_graph_extent_source", "full_physical_record",
                 "gpu_dynamic_graph_extent_source")):
            with self.subTest(field=field):
                self.temporary.cleanup()
                self.temporary = tempfile.TemporaryDirectory()
                self.root = Path(self.temporary.name)
                for mode in analyzer.MODES:
                    report = make_triplet_report(mode)
                    if mode == "dynaextent":
                        report["meta"][field] = value
                    self.write_report(mode, document=report)
                with self.assertRaisesRegex(analyzer.ReportError, message):
                    analyzer.discover_triplets(self.root)

    def test_rejects_missing_or_unsuccessful_reset_log(self):
        self.write_triplet()
        reset_log = (
            self.root / "dynaextent" / "concurrency_336" /
            "repeat_1" / analyzer.RESET_LOG_NAME)
        reset_log.unlink()
        with self.assertRaisesRegex(analyzer.ReportError, "reset log"):
            analyzer.discover_triplets(self.root)

    def test_rejects_initial_recall_or_completed_write_mismatch(self):
        for mutation, message in (
                ("recall", "initial_recall"),
                ("writes", "warmup_completed_writes")):
            with self.subTest(mutation=mutation):
                self.temporary.cleanup()
                self.temporary = tempfile.TemporaryDirectory()
                self.root = Path(self.temporary.name)
                for mode in analyzer.MODES:
                    report = make_triplet_report(mode)
                    if mode == "dynaextent" and mutation == "recall":
                        report["recall"]["recall"] -= 0.001
                    if mode == "dynaextent" and mutation == "writes":
                        phase = report["meta"]["warmup_mixed"]
                        for field in (
                                "issued_writes", "completed_writes",
                                "issued_inserts", "completed_inserts"):
                            phase[field] -= 1
                    self.write_report(mode, document=report)
                with self.assertRaisesRegex(analyzer.ReportError, message):
                    analyzer.discover_triplets(self.root)

    def test_rejects_missing_noninteger_or_inconsistent_raw_telemetry(self):
        mutations = (
            ("missing", "missing JSON field"),
            ("float", "is not an integer"),
            ("fallback", "fallback reads"),
        )
        for mutation, message in mutations:
            with self.subTest(mutation=mutation):
                self.temporary.cleanup()
                self.temporary = tempfile.TemporaryDirectory()
                self.root = Path(self.temporary.name)
                for mode in analyzer.MODES:
                    report = make_triplet_report(mode)
                    if mode == "dynaextent":
                        gpu = report["gpu_persistent"]
                        if mutation == "missing":
                            gpu.pop("dynamic_graph_hint_demotions")
                        elif mutation == "float":
                            gpu["dynamic_graph_hint_demotions"] = 1.0
                        else:
                            gpu["dynamic_graph_fallback_reads"] = (
                                gpu["dynamic_graph_full_reads"] + 1)
                    self.write_report(mode, document=report)
                with self.assertRaisesRegex(analyzer.ReportError, message):
                    analyzer.discover_triplets(self.root)

    def test_rejects_tampered_physical_derived_metric(self):
        for mode in analyzer.MODES:
            report = make_triplet_report(mode)
            if mode == "dynaextent":
                report["gpu_persistent"][
                    "dynamic_graph_short_physical_ratio"] += 0.01
            self.write_report(mode, document=report)
        with self.assertRaisesRegex(
                analyzer.ReportError, "does not match derived value"):
            analyzer.discover_triplets(self.root)

    def test_rejects_disabled_short_reads_or_unexercised_dynaextent(self):
        cases = ("disabled", "unexercised")
        for case in cases:
            with self.subTest(case=case):
                self.temporary.cleanup()
                self.temporary = tempfile.TemporaryDirectory()
                self.root = Path(self.temporary.name)
                for mode in analyzer.MODES:
                    report = make_triplet_report(mode)
                    if case == "disabled" and mode == "static-only":
                        _set_dynamic_telemetry(
                            report,
                            short_reads=1,
                            full_reads=DYNAMIC_LOGICAL_READS - 1,
                            read_bytes=(
                                SHORT_RECORD_BYTES +
                                (DYNAMIC_LOGICAL_READS - 1) *
                                FULL_RECORD_BYTES),
                        )
                    if case == "unexercised" and mode == "dynaextent":
                        _set_dynamic_telemetry(
                            report,
                            short_reads=0,
                            full_reads=DYNAMIC_FALLBACKS +
                            DYNAMIC_EXTRA_FULL_RETRIES,
                            read_bytes=(
                                (DYNAMIC_FALLBACKS +
                                 DYNAMIC_EXTRA_FULL_RETRIES) *
                                FULL_RECORD_BYTES),
                        )
                    self.write_report(mode, document=report)
                expected = (
                    "disables DynaExtent" if case == "disabled" else
                    "enabled but no dynamic short")
                with self.assertRaisesRegex(analyzer.ReportError, expected):
                    analyzer.discover_triplets(self.root)


if __name__ == "__main__":
    unittest.main()
