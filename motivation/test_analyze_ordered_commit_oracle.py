#!/usr/bin/env python3

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_ordered_commit_oracle as oracle


def metadata():
    return {
        "type": "metadata",
        "schema": 3,
        "completion_granularity": "parent_read_cqe_probe",
        "timestamp_clock": "GPU globaltimer nanoseconds",
        "natural_parent_tile": 4,
    }


def query(reads=2, service_ns=20_000):
    # 1 GHz makes one cycle one nanosecond.
    return {
        "type": "query",
        "request_id": 1,
        "status": 0,
        "event_count": reads,
        "overflow": 0,
        "gpu_clock_khz": 1_000_000,
        "gpu_cycles": 100_000,
        "rdma_wait_cycles": 50_000,
        "graph_validation_cycles": service_ns,
        "neighbor_decode_cycles": 0,
        "pq_score_cycles": 0,
        "visited_cycles": 0,
        "graph_rounds": 1,
        "graph_reads": reads,
    }


def parent_event(parent, completion_ns, parents=1):
    return {
        "type": "parent_read",
        "request_id": 1,
        "route_attempt": 0,
        "search_round": 0,
        "snapshot_attempt": 0,
        "target_shard": 0,
        "parent_ordinal": parent,
        "parent_count": parents,
        "bytes_per_parent": 832,
        "issue_timestamp_ns": 1_000,
        "wait_phase_start_timestamp_ns": 2_000,
        "completion_timestamp_ns": completion_ns,
        "batch_process_start_timestamp_ns": 21_000,
    }


class OrderedCommitOracleTest(unittest.TestCase):
    def run_trace(self, records, overheads=(0.0,), granularity="completion"):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8")
            return oracle.analyze(
                path, overheads, granularity, include_rounds=True)

    def test_single_release_has_no_strict_opportunity(self):
        records = [
            metadata(),
            query(reads=1, service_ns=10_000),
            parent_event(0, 10_000),
        ]
        result = self.run_trace(records)
        row = result["rounds"][0]
        self.assertEqual(row["strict_completion_spread_ns"], 0)
        self.assertEqual(
            row["strict_saved_ns_by_task_overhead"]["0us"], 0)

    def test_long_early_task_hides_complete_tail(self):
        # Each parent carries 10 us of modeled service. Parent 0 is released
        # 10 us before parent 1, so its complete service can hide the tail.
        records = [
            metadata(),
            query(reads=2, service_ns=20_000),
            parent_event(0, 10_000),
            parent_event(1, 20_000),
        ]
        result = self.run_trace(records)
        row = result["rounds"][0]
        self.assertEqual(row["strict_completion_spread_ns"], 10_000)
        self.assertEqual(
            row["strict_saved_ns_by_task_overhead"]["0us"], 10_000)

    def test_short_early_task_only_hides_its_service(self):
        # 8 us total -> 4 us per parent. Only 4 us of the 10 us tail can be
        # used by a serial query worker.
        records = [
            metadata(),
            query(reads=2, service_ns=8_000),
            parent_event(0, 10_000),
            parent_event(1, 20_000),
        ]
        result = self.run_trace(records)
        row = result["rounds"][0]
        self.assertEqual(
            row["strict_saved_ns_by_task_overhead"]["0us"], 4_000)

    def test_task_overhead_is_not_hidden_from_screen(self):
        records = [
            metadata(),
            query(reads=2, service_ns=20_000),
            parent_event(0, 10_000),
            parent_event(1, 20_000),
        ]
        result = self.run_trace(records, overheads=(0.0, 6.0))
        row = result["rounds"][0]
        self.assertEqual(
            row["strict_saved_ns_by_task_overhead"]["0us"], 10_000)
        self.assertEqual(
            row["strict_saved_ns_by_task_overhead"]["6us"], 0)

    def test_release_is_clamped_to_common_wait_start(self):
        first = parent_event(0, 1_500)
        first["wait_phase_start_timestamp_ns"] = 2_000
        first["issue_timestamp_ns"] = 1_000
        second = parent_event(1, 5_000)
        records = [
            metadata(),
            query(reads=2, service_ns=2_000),
            first,
            second,
        ]
        result = self.run_trace(records)
        row = result["rounds"][0]
        self.assertEqual(row["release_min_ns"], 2_000)
        self.assertEqual(row["strict_completion_spread_ns"], 3_000)


if __name__ == "__main__":
    unittest.main()
