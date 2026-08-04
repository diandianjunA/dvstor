import copy
import json
import tempfile
import unittest
from pathlib import Path

import motivation.summarize_live_extent_ab as analyzer


def make_report(policy, concurrency, multiplier=1.0):
    completed = 100
    graph_reads = 1000
    graph_rounds = 200
    graph_bytes = (
        832_000 if policy == "fixed" else int(400_000 * multiplier))
    total_bytes = (
        1_000_000 if policy == "fixed" else int(568_000 * multiplier))
    short_reads = 0 if policy == "fixed" else 900
    full_reads = 1000 if policy == "fixed" else 110
    fallback_reads = 0 if policy == "fixed" else 10
    latency_multiplier = 1.0 if policy == "fixed" else 0.9 * multiplier
    qps_multiplier = 1.0 if policy == "fixed" else 1.1 / multiplier
    recall = 0.94
    recall_section = {
        "base_id_limit": 0,
        "groundtruth_file": "/index/groundtruth.bin",
        "k": 10,
        "mode": "all",
        "phase": "before_performance",
        "queries": 1000,
        "queries_with_insufficient_base_results": 0,
        "query_file": "/index/query.u8bin",
        "recall": recall,
        "result_set_complete": True,
        "search_result_width": 10,
    }
    post_recall = copy.deepcopy(recall_section)
    post_recall["phase"] = "after_performance"
    return {
        "meta": {
            "candidate_vector_rdma_bytes": 128,
            "client_threads": concurrency,
            "dim": 128,
            "entry_seed_capacity": 4,
            "entry_seed_policy": "nearest",
            "entry_seed_shards": 1,
            "final_rerank_width": 128,
            "fine_grained_breakdown_enabled": True,
            "gpu_graph_prefetch_depth": 16,
            "gpu_graph_entry_capacity": 100,
            "gpu_graph_extent_quantum_edges": 8,
            "gpu_graph_extent_sidecar_format":
                "global_ordinal_u8_gextent8_v1",
            "gpu_graph_physical_record_bytes": 832,
            "gpu_query_beam_merge_policy": "stable-run",
            "gpu_query_expansion_policy": "fixed",
            "gpu_query_graph_read_policy": policy,
            "gpu_query_slots": 256,
            "gpu_rdma_qps": 32,
            "index_prefix": "/index/sift100m",
            "max_expansions": 384,
            "measure_ops": 1000,
            "measure_seconds": 120,
            "mixed_dispatch_policy": "fixed_threads",
            "navigation_quantizer": "opq_pq",
            "performance_query": {
                "canonical_source": "/data/performance.u8bin",
                "data_type": "uint8",
                "row_reuse_policy": "single_pass_no_reuse",
                "rows": 10_000_000,
                "vector_bytes": 128,
            },
            "recall_base_id_limit": 0,
            "recall_only": False,
            "recall_mode": "all",
            "recall_query": {
                "data_type": "uint8",
                "rows": 10_000,
                "source": "/index/query.u8bin",
                "vector_bytes": 128,
            },
            "run_mode": "time",
            "read_ratio": 0.5,
            "search": "gpu_persistent_opq_pq",
            "target_query_qps": 0.0,
            "target_write_qps": 0.0,
            "time_completion_policy": "drain",
            "time_issue_policy":
                "fixed_read_write_threads_until_deadline",
            "traversal_beam_width": 128,
            "vector_bytes": 128,
            "vector_component_size": 1,
            "vector_data_type": "uint8",
            "warmup_ops": 100,
            "warmup_seconds": 30,
            "workload": "query",
            "write_delete_ratio": 0.0,
            "write_insert_ratio": 1.0,
            "write_upsert_ratio": 0.0,
        },
        "gpu_persistent": {
            "average_gpu_query_us": 5000.0 * latency_multiplier,
            "average_gpu_rdma_wait_us": 800.0 * latency_multiplier,
            "average_graph_read_bytes_per_query": graph_bytes / completed,
            "average_graph_rounds_per_query": graph_rounds / completed,
            "centroid_route_query_timeouts": 0,
            "direct_path_failures": 0,
            "graph_dependency_rounds": graph_rounds,
            "graph_extent_fallback_reads": fallback_reads,
            "graph_full_record_reads": full_reads,
            "graph_live_extent_reads": short_reads,
            "graph_page_requests": graph_reads,
            "graph_read_bytes": graph_bytes,
            "queries_completed": completed,
            "queries_submitted": completed,
            "rdma_read_bytes": total_bytes,
        },
        "query_breakdown": {
            "count": completed,
            "latency": {
                "mean_end_to_end_ns": 5_000_000 * latency_multiplier,
                "p50_end_to_end_ns": 4_900_000 * latency_multiplier,
                "p95_end_to_end_ns": 5_500_000 * latency_multiplier,
                "p99_end_to_end_ns": 6_000_000 * latency_multiplier,
                "p999_end_to_end_ns": 7_000_000 * latency_multiplier,
            },
        },
        "recall": recall_section,
        "stage2": {"failures": 0},
        "static_gt_post_recall": post_recall,
        "throughput": {
            "query_ops": completed,
            "query_ops_per_sec": 20_000.0 * qps_multiplier,
        },
    }


class LiveExtentAbAnalyzerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def write_report(
            self, policy, concurrency, repeat, *, multiplier=1.0,
            document=None):
        directory = (
            self.root / policy / f"concurrency_{concurrency}" /
            f"repeat_{repeat}" / "04_gpu_persistent_gpunetio")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"sift100m_{policy}_{repeat}.json"
        report = (
            make_report(policy, concurrency, multiplier)
            if document is None else document)
        path.write_text(json.dumps(report), encoding="utf-8")
        return path

    def test_pairs_repeats_and_emits_policy_and_paired_medians(self):
        for repeat, multiplier in ((1, 1.0), (2, 1.2), (3, 0.8)):
            self.write_report("fixed", 256, repeat)
            self.write_report(
                "live-extent", 256, repeat, multiplier=multiplier)

        pairs = analyzer.discover_pairs(self.root)
        self.assertEqual(len(pairs), 3)
        summary = analyzer.build_summary(self.root, pairs)
        case = summary["cases"]["concurrency_256"]
        self.assertEqual(case["repeat_count"], 3)
        self.assertEqual(
            case["policy_medians"]["fixed"][
                "actual_graph_bytes_per_query"],
            8320.0,
        )
        self.assertEqual(
            case["policy_medians"]["live-extent"][
                "actual_graph_bytes_per_query"],
            4000.0,
        )
        self.assertAlmostEqual(
            case["paired_medians"]["qps_ratio_live_over_fixed"],
            1.1,
        )
        self.assertAlmostEqual(
            case["paired_medians"]["graph_bytes_reduction_fraction"],
            1.0 - 400.0 / 832.0,
        )
        self.assertEqual(
            [entry["repeat"] for entry in case["repeats"]],
            [1, 2, 3],
        )
        markdown = analyzer.render_markdown(summary)
        self.assertIn("Every paired repeat", markdown)
        self.assertIn("total_rdma_bytes_reduction_fraction", markdown)

        json_path = self.root / "out.json"
        markdown_path = self.root / "out.md"
        result = analyzer.main([
            str(self.root),
            "--json-output", str(json_path),
            "--markdown-output", str(markdown_path),
        ])
        self.assertEqual(result, 0)
        self.assertTrue(json_path.is_file())
        self.assertTrue(markdown_path.is_file())

    def test_rejects_policy_metadata_mismatch(self):
        self.write_report("fixed", 64, 1)
        bad = make_report("fixed", 64)
        self.write_report("live-extent", 64, 1, document=bad)
        with self.assertRaisesRegex(
                analyzer.ReportError,
                "gpu_query_graph_read_policy='fixed', expected 'live-extent'"):
            analyzer.discover_pairs(self.root)

    def test_removed_expansion_policy_field_is_optional_and_ignored(self):
        fixed = make_report("fixed", 64)
        live = make_report("live-extent", 64)
        live["meta"].pop("gpu_query_expansion_policy")
        self.write_report("fixed", 64, 1, document=fixed)
        self.write_report("live-extent", 64, 1, document=live)

        pairs = analyzer.discover_pairs(self.root)
        self.assertEqual(len(pairs), 1)

    def test_rejects_missing_paired_case(self):
        self.write_report("fixed", 1, 1)
        self.write_report("live-extent", 1, 1)
        self.write_report("fixed", 256, 2)
        with self.assertRaisesRegex(
                analyzer.ReportError,
                "unpaired case concurrency=256, repeat=2"):
            analyzer.discover_pairs(self.root)


if __name__ == "__main__":
    unittest.main()
