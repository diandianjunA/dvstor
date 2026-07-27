#!/usr/bin/env python3

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_live_extent_motivation as oracle


class LiveExtentMotivationTest(unittest.TestCase):
    def write_jsonl(self, records):
        temporary = tempfile.TemporaryDirectory()
        path = Path(temporary.name) / "trace.jsonl"
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8")
        self.addCleanup(temporary.cleanup)
        return path

    @staticmethod
    def metadata():
        return {
            "type": "adjacency_oracle_metadata",
            "graph_entry_bytes": 832,
            "remote_ptr_bytes": 8,
        }

    @staticmethod
    def event():
        # This is the aggregate of three parents with degrees 3, 9 and 17.
        return {
            "type": "adjacency_oracle",
            "request_id": 1,
            "search_round": 0,
            "parent_count": 3,
            "edge_count": 29,
            "total_groups": [9, 6, 4, 3],
            "parents_with_tail": [2, 1, 0, 0, 0],
            "total_tail_edges": [10, 1, 0, 0, 0],
        }

    def test_exact_live_and_extent_byte_models(self):
        records = [
            self.metadata(),
            {
                "type": "query",
                "request_id": 1,
                "adjacency_oracle_overflow": 0,
            },
            self.event(),
        ]
        metadata, queries, events = oracle.load_jsonl(
            self.write_jsonl(records))
        report = oracle.aggregate(metadata, queries, events)

        self.assertEqual(report["fixed_record"]["payload_bytes"], 3 * 832)
        self.assertEqual(
            report["ideal_live_prefix"]["payload_bytes"],
            3 * 16 + 29 * 8)
        self.assertEqual(report["extent_8"]["extent_count"], 6)
        self.assertEqual(
            report["extent_8"]["payload_bytes"],
            3 * 16 + 6 * 8 * 8)

        prefix8 = report["continuation"][0]
        self.assertEqual(prefix8["parents_with_continuation"], 2)
        self.assertEqual(prefix8["live_tail_edges"], 10)
        self.assertEqual(prefix8["contiguous_tail_wqes"], 5)
        self.assertEqual(prefix8["extent_tail_wqes_lower"], 2)
        self.assertEqual(prefix8["extent_tail_wqes_upper"], 3)

    def test_projects_graph_ratio_into_total_rdma_bytes(self):
        records = [self.metadata(), self.event()]
        metadata, queries, events = oracle.load_jsonl(
            self.write_jsonl(records))
        # Three fixed graph records plus 1,000 non-graph bytes/query.
        baseline = {
            "query_count": 1,
            "graph_reads_per_query": 3,
            "rdma_ops_per_query": 3,
            "rdma_bytes_per_query": 3 * 832 + 1000,
        }
        report = oracle.aggregate(
            metadata, queries, events, baseline=baseline)
        extent = report["total_rdma_projection"]["extent_8"]
        self.assertEqual(extent["proposed_graph_bytes_per_query"], 432)
        self.assertEqual(
            extent["proposed_total_rdma_bytes_per_query"], 1432)
        self.assertAlmostEqual(
            extent["total_rdma_byte_reduction"],
            1.0 - 1432 / (3 * 832 + 1000))

    def test_rejects_fabricated_non_monotone_prefix_statistics(self):
        event = self.event()
        event["parents_with_tail"] = [1, 2, 0, 0, 0]
        metadata, queries, events = oracle.load_jsonl(
            self.write_jsonl([self.metadata(), event]))
        with self.assertRaisesRegex(ValueError, "non-monotone"):
            oracle.aggregate(metadata, queries, events)


if __name__ == "__main__":
    unittest.main()
