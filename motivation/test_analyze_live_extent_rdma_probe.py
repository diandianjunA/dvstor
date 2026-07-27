#!/usr/bin/env python3

import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_live_extent_rdma_probe as analyzer


class LiveExtentRdmaProbeAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "live_extent_rdma.csv"

    @staticmethod
    def make_row(
        repeat,
        stage1,
        stage2,
        read_wqe_rate,
        p50,
        p99,
        *,
        qps=1,
        batch_reads=16,
        measured_batches=10,
    ):
        stages = 1 if stage2 == 0 else 2
        payload = stage1 + stage2
        read_wqes = qps * measured_batches * batch_reads * stages
        elapsed_ms = read_wqes / read_wqe_rate * 1000.0
        cqes = qps * measured_batches * stages
        application_gb_s = (
            qps * measured_batches * batch_reads * payload
            / (elapsed_ms / 1000.0)
            / 1.0e9
        )
        return {
            "repeat": repeat,
            "order": "reverse" if repeat % 2 == 0 else "forward",
            "stage1_B": stage1,
            "stage2_B": stage2,
            "payload_B": payload,
            "stages": stages,
            "active_QPs": qps,
            "batch_reads": batch_reads,
            "working_set_B": 64 * 1024 * 1024,
            "measured_batches_per_QP": measured_batches,
            "read_WQEs": read_wqes,
            "dump_WQEs": 0,
            "transport_WQEs": read_wqes,
            "CQEs": cqes,
            "elapsed_ms": elapsed_ms,
            "read_WQE_per_s": read_wqe_rate,
            "application_payload_GB_per_s": application_gb_s,
            "batch_latency_mean_us": p50 * 1.05,
            "batch_latency_p50_us": p50,
            "batch_latency_p95_us": p99 * 0.9,
            "batch_latency_p99_us": p99,
        }

    def write_rows(self, rows):
        with self.path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=analyzer.REQUIRED_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    def complete_rows(self):
        rows = []
        # The repeat multipliers make the paired ratios exact while still
        # exercising median, inclusive-IQR, and sample-CV calculations.
        multipliers = (0.9, 1.0, 1.1)
        for repeat, multiplier in enumerate(multipliers, 1):
            for payload in (400, 448, 832):
                base_rate = {
                    400: 1_200_000.0,
                    448: 1_100_000.0,
                    832: 1_000_000.0,
                }[payload]
                base_p50 = {400: 8.0, 448: 8.5, 832: 10.0}[payload]
                base_p99 = {400: 16.0, 448: 17.0, 832: 20.0}[payload]
                rows.append(self.make_row(
                    repeat,
                    payload,
                    0,
                    base_rate * multiplier,
                    base_p50 * multiplier,
                    base_p99 * multiplier,
                ))
            rows.append(self.make_row(
                repeat,
                16,
                400,
                1_800_000.0 * multiplier,
                14.0 * multiplier,
                28.0 * multiplier,
            ))
            rows.append(self.make_row(
                repeat,
                16,
                448,
                1_650_000.0 * multiplier,
                15.3 * multiplier,
                30.6 * multiplier,
            ))
        return rows

    def test_aggregates_repeats_and_builds_paired_comparisons(self):
        self.write_rows(self.complete_rows())
        report = analyzer.analyze(analyzer.load_csv(self.path))

        groups = {
            (
                group["active_QPs"],
                group["stages"],
                group["stage1_B"],
                group["stage2_B"],
            ): group
            for group in report["groups"]
        }
        wqe_summary = groups[(1, 1, 832, 0)]["metrics"][
            "read_WQE_per_s"]
        self.assertEqual(wqe_summary["count"], 3)
        self.assertEqual(wqe_summary["median"], 1_000_000.0)
        self.assertEqual(wqe_summary["iqr"], 100_000.0)
        self.assertAlmostEqual(wqe_summary["cv"], 0.1)

        one_shot = {
            row["candidate_payload_B"]: row
            for row in report["comparisons"][
                "one_shot_400_448_vs_832"]
        }
        self.assertAlmostEqual(
            one_shot[400]["metrics"]["read_WQE_per_s"][
                "paired_ratio"]["median"],
            1.2,
        )
        self.assertAlmostEqual(
            one_shot[400]["metrics"]["batch_latency_p50_us"][
                "paired_ratio"]["median"],
            0.8,
        )
        self.assertAlmostEqual(
            one_shot[448]["metrics"]["batch_latency_p99_us"][
                "paired_ratio"]["median"],
            0.85,
        )

        dependent = {
            row["body_B"]: row
            for row in report["comparisons"][
                "dependent_16_plus_body_vs_corresponding_one_shot"]
        }
        self.assertEqual(
            dependent[400]["metrics"]["read_WQEs"][
                "paired_ratio"]["median"],
            2.0,
        )
        self.assertAlmostEqual(
            dependent[400]["metrics"]["batch_latency_p50_us"][
                "paired_ratio"]["median"],
            1.75,
        )
        self.assertAlmostEqual(
            dependent[400]["metrics"]["batch_latency_p50_us"][
                "paired_delta"]["median"],
            6.0,
        )
        self.assertEqual(
            report["scope"]["classification"],
            "transport-only GPU-initiated one-sided RDMA READ microbenchmark",
        )

    def test_rejects_missing_repeat_in_one_group(self):
        rows = self.complete_rows()
        rows = [
            row for row in rows
            if not (
                row["repeat"] == 3
                and row["stage1_B"] == 448
                and row["stage2_B"] == 0
            )
        ]
        self.write_rows(rows)
        with self.assertRaisesRegex(
                analyzer.ProbeAnalysisError, "expected \\[1, 2, 3\\]"):
            analyzer.analyze(analyzer.load_csv(self.path))

    def test_rejects_inconsistent_wqe_count(self):
        rows = self.complete_rows()
        rows[0]["read_WQEs"] += 1
        rows[0]["transport_WQEs"] += 1
        self.write_rows(rows)
        with self.assertRaisesRegex(
                analyzer.ProbeAnalysisError, "read_WQEs=.*expected"):
            analyzer.load_csv(self.path)

    def test_markdown_states_transport_only_boundary(self):
        self.write_rows(self.complete_rows())
        report = analyzer.analyze(analyzer.load_csv(self.path))
        rendered = analyzer.markdown(report)
        self.assertIn("Scope: transport-only", rendered)
        self.assertIn("does not measure query QPS", rendered)
        self.assertIn("Dependent 16B header", rendered)


if __name__ == "__main__":
    unittest.main()
