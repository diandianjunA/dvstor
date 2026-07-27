#!/usr/bin/env python3

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import summarize_live_extent_motivation as summary


class LiveExtentSummaryTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def write_json(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def extent(self):
        return {
            "payload_ratio": 0.5,
            "record_bytes": 800.0,
            "bytes_per_parent": 400.0,
        }

    def stable_run(self, concurrency=8):
        # 10 queries, two graph reads/query, 2,000 total bytes/query:
        # graph is 80% of all RDMA bytes. Halving graph bytes therefore
        # reduces total bytes by 40%, yielding a 1/0.6 bandwidth roofline.
        return self.write_json(
            f"stable-run/concurrency_{concurrency}/result.json",
            {
                "throughput": {
                    "effective_query_ops_per_sec": 1000,
                },
                "query_breakdown": {"count": 10},
                "gpu_persistent": {
                    "queries_completed": 10,
                    "rdma_read_bytes": 20_000,
                    "graph_page_requests": 20,
                    "rdma_read_ops": 30,
                },
            })

    def test_byte_fraction_and_bandwidth_only_roofline(self):
        report = summary.summarize(self.extent(), [self.stable_run()])
        row = report["runs"][0]
        self.assertEqual(row["concurrency"], 8)
        self.assertEqual(row["current_total_bytes_per_query"], 2000)
        self.assertEqual(row["current_graph_bytes_per_query"], 1600)
        self.assertAlmostEqual(row["current_graph_byte_fraction"], 0.8)
        self.assertEqual(row["extent_total_bytes_per_query"], 1200)
        self.assertAlmostEqual(row["extent_total_byte_reduction"], 0.4)
        self.assertAlmostEqual(
            row["bandwidth_only_speedup_upper_bound"], 5 / 3)
        self.assertAlmostEqual(
            row["bandwidth_only_qps_upper_bound_at_observed_byte_rate"],
            5000 / 3)
        self.assertAlmostEqual(row["current_total_GB_per_s"], 0.002)

    def test_orders_concurrencies_and_rejects_duplicates(self):
        c64 = self.stable_run(64)
        c1 = self.stable_run(1)
        report = summary.summarize(self.extent(), [c64, c1])
        self.assertEqual(
            [row["concurrency"] for row in report["runs"]], [1, 64])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summary.summarize(self.extent(), [c1, c1])

    def test_load_extent_rejects_non_oracle_json(self):
        path = self.write_json("extent.json", {"extent_8": {}})
        with self.assertRaisesRegex(ValueError, "live-extent"):
            summary.load_extent(path)


if __name__ == "__main__":
    unittest.main()
