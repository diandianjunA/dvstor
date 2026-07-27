#!/usr/bin/env python3
import json
import tempfile
import unittest
from pathlib import Path

from analyze_feedback_pricing_motivation import analyze


class FeedbackPricingAnalysisTest(unittest.TestCase):
    def test_tracks_origin_productivity_and_suffix_oracle(self):
        records = [
            {
                "type": "query", "request_id": 7, "gpu_cycles": 1000,
            },
            {
                "type": "adjacency_oracle", "schema": 2, "request_id": 7,
                "search_round": 0, "chunk_begin": 0, "parent_count": 4,
                "selected_handles": [1, 2, 3, 4],
                "selected_productive_mask": 0b0011,
                "frontier_count": 4,
                "frontier_handles": [10, 11, 12, 13],
                "frontier_new_mask": 0b0011,
                "round_graph_cycles": 100, "round_score_cycles": 100,
                "round_beam_cycles": 50,
            },
            {
                "type": "adjacency_oracle", "schema": 2, "request_id": 7,
                "search_round": 1, "chunk_begin": 0, "parent_count": 2,
                "selected_handles": [10, 12],
                "selected_productive_mask": 0b0001,
                "frontier_count": 2,
                "frontier_handles": [20, 21],
                "frontier_new_mask": 0b0001,
                "round_graph_cycles": 50, "round_score_cycles": 50,
                "round_beam_cycles": 25,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            result = analyze(path)
        self.assertEqual(result["sampled_queries"], 1)
        self.assertEqual(result["sampled_rounds"], 2)
        self.assertGreater(
            result["productive_suffix_oracle"]["removable_parent_fraction"], 0
        )
        selected = {
            (row["origin"], row["turnover_bin"], row["rank_band"]): row
            for row in result["frontier_eventual_selection"]
        }
        self.assertEqual(
            selected[("new", "[.50,.75)", "0-15")]["positive"], 1
        )


if __name__ == "__main__":
    unittest.main()
