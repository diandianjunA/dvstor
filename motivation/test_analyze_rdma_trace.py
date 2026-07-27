#!/usr/bin/env python3

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("analyze_rdma_trace.py")
SPEC = importlib.util.spec_from_file_location("analyze_rdma_trace", MODULE_PATH)
ANALYZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ANALYZER)


class AnalyzeRdmaTraceTest(unittest.TestCase):
    def write_trace(self, records):
        temporary = tempfile.TemporaryDirectory()
        path = Path(temporary.name) / "trace.jsonl"
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8")
        self.addCleanup(temporary.cleanup)
        return path

    def test_separates_straggler_barrier_from_handoff(self):
        records = [
            {
                "type": "metadata",
                "schema": 2,
                "completion_granularity":
                    "shard_batch_owner_completion_boundary",
                "natural_parent_tile": 4,
            },
            {
                "type": "query",
                "request_id": 1,
                "status": 0,
                "event_count": 3,
                "overflow": 0,
                "gpu_cycles": 1000,
                "gpu_clock_khz": 1_000_000,
                "rdma_wait_cycles": 500,
            },
            {
                "type": "shard_batch",
                "request_id": 1,
                "route_attempt": 0,
                "search_round": 0,
                "snapshot_attempt": 0,
                "target_shard": 0,
                "parent_count": 3,
                "bytes_per_parent": 128,
                "issue_timestamp_ns": 100,
                "wait_phase_start_timestamp_ns": 150,
                "completion_timestamp_ns": 200,
                "batch_process_start_timestamp_ns": 410,
            },
            {
                "type": "shard_batch",
                "request_id": 1,
                "route_attempt": 0,
                "search_round": 0,
                "snapshot_attempt": 0,
                "target_shard": 1,
                "parent_count": 1,
                "bytes_per_parent": 128,
                "issue_timestamp_ns": 120,
                "wait_phase_start_timestamp_ns": 150,
                "completion_timestamp_ns": 400,
                "batch_process_start_timestamp_ns": 410,
            },
            {
                "type": "shard_batch",
                "request_id": 1,
                "route_attempt": 0,
                "search_round": 1,
                "snapshot_attempt": 0,
                "target_shard": 0,
                "parent_count": 4,
                "payload_bytes": 448,
                "minimum_bytes_per_parent": 80,
                "maximum_bytes_per_parent": 144,
                "issue_timestamp_ns": 500,
                "wait_phase_start_timestamp_ns": 510,
                "completion_timestamp_ns": 600,
                "batch_process_start_timestamp_ns": 605,
            },
        ]
        summary = ANALYZER.analyze_trace(self.write_trace(records))
        rounds = summary["round_attempts"]
        multi = next(row for row in rounds if row["search_round"] == 0)
        single = next(row for row in rounds if row["search_round"] == 1)

        self.assertEqual(multi["completion_max_minus_min_ns"], 200)
        self.assertEqual(multi["strict_wait_completion_spread_ns"], 200)
        self.assertEqual(multi["post_completion_handoff_ns"], 10)
        self.assertEqual(multi["straggler_barrier_parent_ns"], 600)
        self.assertEqual(multi["strict_wait_barrier_parent_ns"], 600)
        self.assertEqual(multi["ready_until_process_parent_ns"], 640)
        self.assertAlmostEqual(
            multi["normalized_straggler_barrier_waste"], 0.5)
        self.assertAlmostEqual(
            multi["normalized_strict_wait_barrier_waste"], 0.6)
        self.assertAlmostEqual(
            multi["ready_before_tail_parent_fraction"], 0.75)
        self.assertEqual(single["bytes_total"], 448)

        aggregate = summary["aggregate"]["primary_complete"]
        self.assertEqual(aggregate["round_attempts"], 2)
        self.assertEqual(aggregate["multi_shard_round_attempts"], 1)
        self.assertEqual(aggregate["single_shard_round_attempts"], 1)
        self.assertEqual(aggregate["completion_spread_p50_ns"], 200)
        query = summary["queries"][0]
        self.assertEqual(
            query["primary_strict_wait_overlap_window_upper_bound_ns"], 200)
        self.assertAlmostEqual(
            query["strict_wait_overlap_upper_bound_over_rdma_wait"], 0.4)

    def test_clamps_completions_that_arrive_during_issue(self):
        records = [
            {"type": "metadata", "schema": 2},
            {
                "type": "query",
                "request_id": 7,
                "status": 0,
                "event_count": 2,
                "overflow": 0,
                "gpu_cycles": 10,
                "gpu_clock_khz": 1,
                "rdma_wait_cycles": 0,
            },
            {
                "type": "shard_batch",
                "request_id": 7,
                "route_attempt": 0,
                "search_round": 0,
                "snapshot_attempt": 0,
                "target_shard": 0,
                "parent_count": 1,
                "bytes_per_parent": 64,
                "issue_timestamp_ns": 100,
                "wait_phase_start_timestamp_ns": 300,
                "completion_timestamp_ns": 200,
                "batch_process_start_timestamp_ns": 320,
            },
            {
                "type": "shard_batch",
                "request_id": 7,
                "route_attempt": 0,
                "search_round": 0,
                "snapshot_attempt": 0,
                "target_shard": 1,
                "parent_count": 1,
                "bytes_per_parent": 64,
                "issue_timestamp_ns": 110,
                "wait_phase_start_timestamp_ns": 300,
                "completion_timestamp_ns": 250,
                "batch_process_start_timestamp_ns": 320,
            },
        ]
        summary = ANALYZER.analyze_trace(self.write_trace(records))
        row = summary["round_attempts"][0]
        self.assertEqual(row["completion_max_minus_min_ns"], 50)
        self.assertEqual(row["strict_wait_completion_spread_ns"], 0)
        self.assertEqual(row["strict_wait_barrier_parent_ns"], 0)
        self.assertEqual(row["ready_at_wait_start_parent_fraction"], 1.0)

    def test_reports_invalid_overflow_and_retry_without_silent_drop(self):
        records = [
            {"type": "metadata", "schema": 2},
            {
                "type": "query",
                "request_id": 9,
                "status": -5,
                "event_count": 2,
                "overflow": 1,
                "gpu_cycles": 10,
                "gpu_clock_khz": 1,
                "rdma_wait_cycles": 1,
            },
            {
                "type": "shard_batch",
                "request_id": 9,
                "route_attempt": 1,
                "search_round": 0,
                "snapshot_attempt": 1,
                "target_shard": 0,
                "parent_count": 1,
                "bytes_per_parent": 64,
                "issue_timestamp_ns": 100,
                "wait_phase_start_timestamp_ns": 110,
                "completion_timestamp_ns": 130,
                "batch_process_start_timestamp_ns": 140,
            },
            {
                "type": "shard_batch",
                "request_id": 9,
                "route_attempt": 1,
                "search_round": 1,
                "snapshot_attempt": 0,
                "target_shard": 0,
                "parent_count": 1,
                "bytes_per_parent": 64,
                "issue_timestamp_ns": 0,
                "wait_phase_start_timestamp_ns": 0,
                "completion_timestamp_ns": 0,
                "batch_process_start_timestamp_ns": 0,
            },
        ]
        summary = ANALYZER.analyze_trace(self.write_trace(records))
        integrity = summary["integrity"]
        self.assertEqual(integrity["invalid_timestamp_events"], 1)
        self.assertEqual(integrity["incomplete_round_attempt_groups"], 1)
        self.assertEqual(integrity["trace_overflow_queries"], 1)
        self.assertEqual(integrity["failed_queries"], 1)
        self.assertEqual(
            summary["aggregate"]["retry_complete"]["round_attempts"], 1)
        self.assertEqual(len(summary["queries"]), 1)


if __name__ == "__main__":
    unittest.main()
