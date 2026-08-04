import copy
import json
import tempfile
import unittest
from pathlib import Path

import motivation.summarize_live_extent_mixed_ab as analyzer


def _mixed_phase(reads, writes, *, scheduled):
    return {
        "issued_reads": reads,
        "issued_writes": writes,
        "issued_inserts": writes,
        "issued_upserts": 0,
        "issued_deletes": 0,
        "completed_reads": reads,
        "completed_writes": writes,
        "completed_inserts": writes,
        "completed_upserts": 0,
        "completed_deletes": 0,
        "scheduled_reads": reads if scheduled else 0,
        "scheduled_writes": writes if scheduled else 0,
        "drain_seconds": 0.2,
    }


def _recall_section(phase, recall):
    return {
        "base_id_limit": 0,
        "groundtruth_file": "/index/groundtruth.bin",
        "k": 10,
        "mode": "all",
        "phase": phase,
        "queries": 1000,
        "queries_with_insufficient_base_results": 0,
        "query_file": "/index/query.u8bin",
        "recall": recall,
        "result_set_complete": True,
        "search_result_width": 10,
    }


def make_report(policy, contract_name="rate-limited", multiplier=1.0):
    contract = analyzer.CONTRACTS[contract_name]
    rate_limited = contract.mixed_mode == "rate_limited"
    if rate_limited:
        warmup_reads, warmup_writes = 1_200_000, 30_000
        query_count, write_count = 4_800_000, 120_000
    else:
        warmup_reads, warmup_writes = 1_000_000, 30_000
        query_count, write_count = 4_500_000, 120_000
    warmup = _mixed_phase(
        warmup_reads, warmup_writes, scheduled=rate_limited)
    measure = _mixed_phase(
        query_count, write_count, scheduled=rate_limited)

    drain = 0.2 * multiplier
    duration = 120.0 + drain
    maintenance_drain = 0.4 * multiplier
    durable_duration = duration + maintenance_drain
    graph_reads = query_count * 200
    graph_rounds = query_count * 15
    graph_bytes = (
        graph_reads * 832 if policy == "fixed"
        else int(graph_reads * 420 * multiplier))
    short_reads = 0 if policy == "fixed" else graph_reads
    full_reads = graph_reads if policy == "fixed" else 0
    fallback_reads = 0
    exact_reads = query_count * 128
    rdma_wqes = short_reads + full_reads + exact_reads
    total_rdma_bytes = graph_bytes + exact_reads * 128
    latency_factor = 1.0 if policy == "fixed" else 0.9 * multiplier
    gpu_factor = latency_factor

    driver = {
        "semantics": "closed_loop_synchronous_no_drop",
        "client_threads_source": contract.driver_source,
        "selected_client_threads": contract.concurrency,
        "auto_required_threads": contract.driver_required_threads,
        "auto_thread_cap": 1024,
        "auto_cap_applied": False,
        "gpu_query_slot_capacity": 256,
        "storage_rpc_inflight_capacity": 80,
        "storage_shards": 5,
        "storage_rpc_depth_per_shard": 16,
        "derivation": contract.driver_derivation,
    }
    if contract.fixed_read_threads is not None:
        driver["fixed_thread_projected_read_threads"] = 256
        driver["fixed_thread_projected_write_threads"] = 256

    meta = {
        "benchmark_driver_concurrency": driver,
        "candidate_vector_rdma_bytes": 128,
        "client_threads": contract.concurrency,
        "dim": 128,
        "effective_bytes_per_vector": 128,
        "effective_insert_start_id": 111_000_000,
        "entry_seed_capacity": 4,
        "entry_seed_policy": "nearest_centroid_shard_live_entries",
        "entry_seed_shards": 1,
        "final_rerank_width": 128,
        "fine_grained_breakdown_enabled": True,
        "gpu_graph_entry_capacity": 102,
        "gpu_graph_extent_quantum_edges": 8,
        "gpu_graph_extent_sidecar_format":
            "global_ordinal_u8_gextent8_v1",
        "gpu_graph_physical_record_bytes": 832,
        "gpu_graph_prefetch_depth": 16,
        "gpu_query_beam_merge_policy": "stable-run",
        "gpu_query_expansion_policy": "fixed",
        "gpu_query_graph_read_policy": policy,
        "gpu_query_slots": 256,
        "gpu_rdma_qps": 32,
        "index_prefix": "/index/sift100m",
        "insert_start_id": 111_000_000,
        "insert_vector_source": "/data/insert.u8bin",
        "max_expansions": 384,
        "measure_mixed": measure,
        "measure_ops": 1000,
        "measure_seconds": 120,
        "mixed_dispatch_policy": contract.mixed_mode,
        "navigation_quantizer": "opq_pq",
        "node_size": 160,
        "normalized_write_mix": {
            "delete": 0.0,
            "insert": 1.0,
            "upsert": 0.0,
        },
        "operation_granularity": "single_vector",
        "performance_query": {
            "canonical_source": "/data/performance.u8bin",
            "data_type": "uint8",
            "measure_rows_consumed": query_count,
            "rate_limited_required_rows":
                warmup_reads + query_count if rate_limited else 0,
            "remaining_rows": 10_000_000 - warmup_reads - query_count,
            "row_reuse_count": 0,
            "row_reuse_policy": "single_pass_no_reuse",
            "rows": 10_000_000,
            "source": "/data/performance.u8bin",
            "total_rows_consumed": warmup_reads + query_count,
            "vector_bytes": 128,
            "warmup_rows_consumed": warmup_reads,
        },
        "read_ratio": contract.read_ratio,
        "recall_base_id_limit": 0,
        "recall_mode": "all",
        "recall_only": False,
        "recall_query": {
            "data_type": "uint8",
            "purpose": "recall_only",
            "rows": 10_000,
            "source": "/index/query.u8bin",
            "vector_bytes": 128,
        },
        "run_mode": "time",
        "search": "gpu_persistent_opq_pq",
        "target_query_qps": contract.target_query_qps,
        "target_write_qps": contract.target_write_qps,
        "threads": 64,
        "time_completion_policy": "drain",
        "time_issue_policy": contract.time_issue_policy,
        "traversal_beam_width": 128,
        "vector_bytes": 128,
        "vector_component_size": 1,
        "vector_data_type": "uint8",
        "warmup_mixed": warmup,
        "warmup_ops": 100,
        "warmup_seconds": 30,
        "workload": "mixed",
        "write_delete_ratio": 0.0,
        "write_insert_ratio": 1.0,
        "write_upsert_ratio": 0.0,
    }
    if contract.fixed_read_threads is not None:
        meta["mixed_fixed_threads"] = {
            "read_threads": 256,
            "write_threads": 256,
        }

    query_qps = (
        query_count / 120.0 if rate_limited else query_count / duration)
    write_qps = (
        write_count / 120.0 if rate_limited else write_count / duration)
    throughput = {
        "configured_measure_seconds": 120,
        "configured_total_measure_seconds": 120.0,
        "effective_measure_seconds": duration,
        "duration_seconds": duration,
        "client_drain_seconds": drain,
        "query_client_drain_seconds": drain,
        "write_client_drain_seconds": drain,
        "query_duration_seconds": duration,
        "write_duration_seconds": duration,
        "maintenance_drain_seconds": maintenance_drain,
        "durable_effective_measure_seconds": durable_duration,
        "total_ops": query_count + write_count,
        "total_ops_per_sec": (query_count + write_count) / duration,
        "query_ops": query_count,
        "query_ops_per_sec": query_qps,
        "nominal_query_ops_per_sec": query_qps,
        "effective_query_ops_per_sec": query_count / duration,
        "write_ops": write_count,
        "write_ops_per_sec": write_qps,
        "durable_write_ops_per_sec": write_count / durable_duration,
        "nominal_write_ops_per_sec": write_qps,
        "effective_write_ops_per_sec": write_count / duration,
        "insert_ops": write_count,
        "insert_ops_per_sec": write_qps,
        "nominal_insert_ops_per_sec": write_qps,
        "effective_insert_ops_per_sec": write_count / duration,
        "durable_total_ops_per_sec":
            (query_count + write_count) / durable_duration,
        "scheduled_query_ops": query_count if rate_limited else 0,
        "scheduled_write_ops": write_count if rate_limited else 0,
        "query_rate_attainment_ratio": 1.0,
        "write_rate_attainment_ratio": 1.0,
        "nominal_rate_basis":
            "configured_schedule_window" if rate_limited
            else "effective_wall_clock",
        "effective_rate_basis": "wall_clock_including_client_drain",
        "durable_rate_basis":
            "wall_clock_including_client_and_stage2_watermark_drain",
        "upsert_ops": 0,
        "upsert_ops_per_sec": 0.0,
        "delete_ops": 0,
        "delete_ops_per_sec": 0.0,
    }

    gpu = {
        "queries_submitted": query_count,
        "queries_completed": query_count,
        "direct_path_failures": 0,
        "centroid_route_query_timeouts": 0,
        "graph_page_requests": graph_reads,
        "graph_dependency_rounds": graph_rounds,
        "graph_read_bytes": graph_bytes,
        "rdma_read_bytes": total_rdma_bytes,
        "rdma_read_ops": rdma_wqes,
        "graph_live_extent_reads": short_reads,
        "graph_full_record_reads": full_reads,
        "graph_extent_fallback_reads": fallback_reads,
        "graph_extent_fallback_ratio": 0.0,
        "graph_read_retries": 0,
        "average_graph_rounds_per_query": graph_rounds / query_count,
        "average_graph_read_bytes_per_query": graph_bytes / query_count,
        "average_graph_read_bytes_per_logical_parent":
            graph_bytes / graph_reads,
    }
    phase_values = {
        "average_gpu_query_us": 4400.0,
        "average_gpu_prepare_us": 100.0,
        "average_gpu_beam_selection_us": 70.0,
        "average_gpu_rdma_issue_us": 120.0,
        "average_gpu_rdma_wait_us": 800.0,
        "average_gpu_graph_validation_us": 500.0,
        "average_gpu_neighbor_decode_us": 100.0,
        "average_gpu_pq_score_us": 500.0,
        "average_gpu_visited_us": 90.0,
        "average_gpu_beam_merge_us": 1800.0,
        "average_gpu_exact_us": 300.0,
        "average_gpu_other_us": 120.0,
    }
    gpu.update({
        key: value * gpu_factor for key, value in phase_values.items()
    })

    def breakdown(operation, count, mean_ns):
        return {
            "operation": operation,
            "count": count,
            "fine_grained_breakdown_observed": True,
            "latency": {
                "mean_end_to_end_ns": mean_ns,
                "p50_end_to_end_ns": mean_ns * 0.95,
                "p95_end_to_end_ns": mean_ns * 1.15,
                "p99_end_to_end_ns": mean_ns * 1.30,
                "p999_end_to_end_ns": mean_ns * 1.50,
            },
        }

    stage2 = {
        "source": "in_band_control_page",
        "requested_logs": 5,
        "readable_logs": 5,
        "logs_with_observations": 5,
        "logs_with_slope_observations": 5,
        "observations": 10,
        "unreadable_logs": [],
        "remaining": int(20 * multiplier),
        "max_backlog_observed": int(80 * multiplier),
        "backlog_slope_per_sec": 0.5 * multiplier,
        "backlog_slope_available": True,
        "p99_stage2_delay_upper_ms": 1000.0 * multiplier,
        "p99_stage2_delay_over_30s": False,
        "p99_stage2_delay_samples": write_count,
        "p99_stage2_delay_available": True,
        "failures": 0,
        "failure_delta_available": True,
        "peer_reverse_retry_attempts": 2,
        "peer_reverse_retry_delta_available": True,
        "admission_window": 640,
        "completion_outstanding": int(40 * multiplier),
        "max_completion_outstanding_per_shard": 128,
        "completion_window_available": True,
        "locality_delta_available": True,
        "stage2_finalized_live_delta": write_count,
        "stage2_continuations": write_count,
        "stage2_remote_frontier_items": write_count * 100,
        "stage2_batches": write_count // 2,
        "stage2_batched_items": write_count,
        "execution_counter_delta_available": True,
        "pressure_yields": int(1000 * multiplier),
        "search_budget_delta_available": True,
        "stage1_search_budget_exhausted": 0,
        "stage2_search_budget_exhausted": 0,
        "observation_period_seconds_assumed": 5.0,
    }
    storage = {
        "late_rpc_completions": 0,
        "late_rpc_threshold_ms": 30_000,
        "maintenance_drain_seconds": maintenance_drain,
        "maintenance_target_sequences": [24_000] * 5,
        "maintenance_durable_sequences": [24_000] * 5,
        "submitted_batches": 1_000_000,
        "submitted_items": 8_000_000,
        "completed_batches": 1_000_000,
        "completed_items": 8_000_000,
        "completed_rpc_wall_ns": 1_000_000,
        "max_rpc_wall_ns": 10_000,
        "average_submitted_batch_size": 8.0,
        "average_completed_rpc_wall_us": 0.001,
    }
    return {
        "meta": meta,
        "throughput": throughput,
        "query_breakdown": breakdown(
            "query", query_count, 4_500_000 * latency_factor),
        "insert_breakdown": breakdown(
            "insert", write_count, 8_000_000 * multiplier),
        "gpu_persistent": gpu,
        "recall": _recall_section("before_performance", 0.94),
        # A mixed run is allowed to change Recall; this is an outcome.
        "static_gt_post_recall": _recall_section(
            "after_performance", 0.938 - 0.001 * (multiplier - 1.0)),
        "stage2": stage2,
        "storage_owner_runtime": storage,
    }


class LiveExtentMixedAbAnalyzerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def write_report(
            self, policy, repeat, *, contract_name="rate-limited",
            multiplier=1.0, document=None):
        contract = analyzer.CONTRACTS[contract_name]
        directory = (
            self.root / policy /
            f"concurrency_{contract.concurrency}" /
            f"repeat_{repeat}" / "04_gpu_persistent_gpunetio")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"sift100m_{policy}_{repeat}.json"
        report = (
            make_report(policy, contract_name, multiplier)
            if document is None else document)
        path.write_text(json.dumps(report), encoding="utf-8")
        return path

    def test_rate_limited_is_default_and_emits_json_and_markdown(self):
        for repeat, multiplier in ((1, 1.0), (2, 1.02), (3, 0.98)):
            self.write_report("fixed", repeat)
            self.write_report(
                "live-extent", repeat, multiplier=multiplier)

        contract = analyzer.CONTRACTS["rate-limited"]
        pairs = analyzer.discover_pairs(self.root, contract)
        self.assertEqual(len(pairs), 3)
        summary = analyzer.build_summary(self.root, pairs, contract)
        self.assertEqual(summary["experiment_contract"]["client_threads"], 336)
        self.assertEqual(summary["pair_count"], 3)
        self.assertAlmostEqual(
            summary["policy_medians"]["fixed"]["query_qps"], 40_000.0)
        self.assertAlmostEqual(
            summary["policy_medians"]["fixed"]["write_qps"], 1_000.0)
        self.assertGreater(
            summary["paired_medians"]["graph_bytes_reduction_fraction"], 0.45)
        markdown = analyzer.render_markdown(summary)
        self.assertIn("Stage2", markdown.replace("stage2", "Stage2"))
        self.assertIn("rdma_wqes_per_query", markdown)

        json_path = self.root / "out.json"
        markdown_path = self.root / "out.md"
        self.assertEqual(analyzer.main([
            str(self.root),
            "--json-output", str(json_path),
            "--markdown-output", str(markdown_path),
        ]), 0)
        self.assertTrue(json_path.is_file())
        self.assertTrue(markdown_path.is_file())

    def test_fixed_threads_contract_remains_explicitly_supported(self):
        self.write_report("fixed", 1, contract_name="fixed-threads")
        self.write_report("live-extent", 1, contract_name="fixed-threads")
        contract = analyzer.CONTRACTS["fixed-threads"]
        pairs = analyzer.discover_pairs(self.root, contract)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][0].concurrency, 512)

    def test_removed_expansion_policy_field_is_optional_and_ignored(self):
        fixed = make_report("fixed")
        live = make_report("live-extent")
        live["meta"].pop("gpu_query_expansion_policy")
        self.write_report("fixed", 1, document=fixed)
        self.write_report("live-extent", 1, document=live)

        pairs = analyzer.discover_pairs(
            self.root, analyzer.CONTRACTS["rate-limited"])
        self.assertEqual(len(pairs), 1)

    def test_required_query_rows_uses_schedule_not_completion(self):
        report = make_report("fixed")
        report["meta"]["warmup_mixed"]["issued_reads"] -= 4
        report["meta"]["warmup_mixed"]["completed_reads"] -= 4
        report["meta"]["performance_query"]["warmup_rows_consumed"] -= 4
        report["meta"]["performance_query"]["total_rows_consumed"] -= 4
        report["meta"]["performance_query"]["remaining_rows"] += 4
        path = self.write_report("fixed", 1, document=report)
        loaded = analyzer.load_report(
            path, "fixed", 336, 1,
            analyzer.CONTRACTS["rate-limited"])
        self.assertEqual(
            loaded.pair_values["warmup_completed_writes"], 30_000)
        self.assertEqual(
            report["meta"]["performance_query"][
                "rate_limited_required_rows"],
            6_000_000,
        )

    def test_fixed_snapshot_retry_is_an_extra_physical_full_read(self):
        report = make_report("fixed")
        gpu = report["gpu_persistent"]
        gpu["graph_full_record_reads"] += 1
        gpu["graph_read_retries"] = 1
        gpu["graph_read_bytes"] += 832
        gpu["rdma_read_bytes"] += 832
        gpu["rdma_read_ops"] += 1
        query_count = gpu["queries_completed"]
        graph_reads = gpu["graph_page_requests"]
        gpu["average_graph_read_bytes_per_query"] = (
            gpu["graph_read_bytes"] / query_count)
        gpu["average_graph_read_bytes_per_logical_parent"] = (
            gpu["graph_read_bytes"] / graph_reads)
        path = self.write_report("fixed", 1, document=report)
        loaded = analyzer.load_report(
            path, "fixed", 336, 1,
            analyzer.CONTRACTS["rate-limited"])
        self.assertGreater(
            loaded.metrics["graph_read_retries_per_query"], 0.0)

    def test_missing_adaptation_telemetry_is_backward_compatible_zero(self):
        report = make_report("live-extent")
        self.assertNotIn(
            "graph_extent_underhint_reads", report["gpu_persistent"])
        self.assertNotIn(
            "graph_extent_hint_promotions", report["gpu_persistent"])
        path = self.write_report("live-extent", 1, document=report)
        loaded = analyzer.load_report(
            path, "live-extent", 336, 1,
            analyzer.CONTRACTS["rate-limited"])
        self.assertEqual(
            loaded.metrics["underhint_graph_reads_per_query"], 0.0)
        self.assertEqual(
            loaded.metrics["extent_hint_promotions_per_query"], 0.0)
        self.assertEqual(
            loaded.metrics["extent_hint_promotion_rate"], 0.0)

    def test_underhint_and_high_water_promotion_telemetry(self):
        report = make_report("live-extent")
        gpu = report["gpu_persistent"]
        fallback_reads = 100
        promotions = 99
        gpu["graph_extent_fallback_reads"] = fallback_reads
        gpu["graph_extent_underhint_reads"] = fallback_reads
        gpu["graph_extent_hint_promotions"] = promotions
        gpu["graph_read_retries"] = fallback_reads
        gpu["graph_full_record_reads"] += fallback_reads
        gpu["graph_read_bytes"] += fallback_reads * 832
        gpu["rdma_read_bytes"] += fallback_reads * 832
        gpu["rdma_read_ops"] += fallback_reads
        gpu["graph_extent_fallback_ratio"] = (
            fallback_reads / gpu["graph_live_extent_reads"])
        query_count = gpu["queries_completed"]
        graph_reads = gpu["graph_page_requests"]
        gpu["average_graph_read_bytes_per_query"] = (
            gpu["graph_read_bytes"] / query_count)
        gpu["average_graph_read_bytes_per_logical_parent"] = (
            gpu["graph_read_bytes"] / graph_reads)
        path = self.write_report("live-extent", 1, document=report)
        loaded = analyzer.load_report(
            path, "live-extent", 336, 1,
            analyzer.CONTRACTS["rate-limited"])
        self.assertAlmostEqual(
            loaded.metrics["underhint_graph_reads_per_query"],
            fallback_reads / query_count,
        )
        self.assertAlmostEqual(
            loaded.metrics["extent_hint_promotions_per_query"],
            promotions / query_count,
        )
        self.assertAlmostEqual(
            loaded.metrics["extent_hint_promotion_rate"], 0.99)

    def test_fixed_rejects_nonzero_adaptation_telemetry(self):
        report = make_report("fixed")
        report["gpu_persistent"]["graph_extent_underhint_reads"] = 1
        report["gpu_persistent"]["graph_extent_hint_promotions"] = 1
        path = self.write_report("fixed", 1, document=report)
        with self.assertRaisesRegex(
                analyzer.ReportError,
                "fixed policy reported live-extent adaptation"):
            analyzer.load_report(
                path, "fixed", 336, 1,
                analyzer.CONTRACTS["rate-limited"])

    def test_rejects_pair_with_different_completed_update_count(self):
        self.write_report("fixed", 1)
        bad = make_report("live-extent")
        for field in (
                "completed_writes", "issued_writes",
                "completed_inserts", "issued_inserts"):
            bad["meta"]["measure_mixed"][field] -= 1
        bad["insert_breakdown"]["count"] -= 1
        bad["throughput"]["write_ops"] -= 1
        bad["throughput"]["insert_ops"] -= 1
        bad["throughput"]["total_ops"] -= 1
        write_ops = bad["throughput"]["write_ops"]
        total_ops = bad["throughput"]["total_ops"]
        duration = bad["throughput"]["duration_seconds"]
        durable_duration = bad["throughput"][
            "durable_effective_measure_seconds"]
        bad["throughput"]["write_ops_per_sec"] = write_ops / 120.0
        bad["throughput"]["nominal_write_ops_per_sec"] = write_ops / 120.0
        bad["throughput"]["insert_ops_per_sec"] = write_ops / 120.0
        bad["throughput"]["nominal_insert_ops_per_sec"] = write_ops / 120.0
        bad["throughput"]["effective_write_ops_per_sec"] = (
            write_ops / duration)
        bad["throughput"]["effective_insert_ops_per_sec"] = (
            write_ops / duration)
        bad["throughput"]["durable_write_ops_per_sec"] = (
            write_ops / durable_duration)
        bad["throughput"]["total_ops_per_sec"] = total_ops / duration
        bad["throughput"]["durable_total_ops_per_sec"] = (
            total_ops / durable_duration)
        bad["throughput"]["write_rate_attainment_ratio"] = (
            write_ops / bad["throughput"]["scheduled_write_ops"])
        self.write_report("live-extent", 1, document=bad)
        with self.assertRaisesRegex(
                analyzer.ReportError, "measure_completed_writes"):
            analyzer.discover_pairs(
                self.root, analyzer.CONTRACTS["rate-limited"])

    def test_stage2_peak_outstanding_is_an_outcome_not_pair_control(self):
        fixed = make_report("fixed")
        live = make_report("live-extent")
        fixed["stage2"]["max_completion_outstanding_per_shard"] = 4
        live["stage2"]["max_completion_outstanding_per_shard"] = 5
        self.write_report("fixed", 1, document=fixed)
        self.write_report("live-extent", 1, document=live)
        pairs = analyzer.discover_pairs(
            self.root, analyzer.CONTRACTS["rate-limited"])
        self.assertEqual(
            pairs[0][0].metrics[
                "stage2_max_completion_outstanding_per_shard"],
            4.0,
        )
        self.assertEqual(
            pairs[0][1].metrics[
                "stage2_max_completion_outstanding_per_shard"],
            5.0,
        )

    def test_rejects_write_accounting_mismatch(self):
        bad = make_report("fixed")
        bad["throughput"]["write_ops"] += 1
        self.write_report("fixed", 1, document=bad)
        self.write_report("live-extent", 1)
        with self.assertRaisesRegex(
                analyzer.ReportError, "throughput operation accounting"):
            analyzer.discover_pairs(
                self.root, analyzer.CONTRACTS["rate-limited"])

    def test_rejects_non_policy_configuration_change(self):
        self.write_report("fixed", 1)
        bad = make_report("live-extent")
        bad["meta"]["threads"] = 63
        self.write_report("live-extent", 1, document=bad)
        with self.assertRaisesRegex(
                analyzer.ReportError, "meta.threads"):
            analyzer.discover_pairs(
                self.root, analyzer.CONTRACTS["rate-limited"])

    def test_rejects_failure_or_incomplete_stage2_observation(self):
        bad = make_report("fixed")
        bad["stage2"]["failures"] = 1
        self.write_report("fixed", 1, document=bad)
        self.write_report("live-extent", 1)
        with self.assertRaisesRegex(
                analyzer.ReportError, "stage2.failures=1"):
            analyzer.discover_pairs(
                self.root, analyzer.CONTRACTS["rate-limited"])

    def test_post_recall_is_outcome_but_initial_recall_is_certificate(self):
        fixed = make_report("fixed")
        live = make_report("live-extent")
        live["static_gt_post_recall"]["recall"] = 0.931
        self.write_report("fixed", 1, document=fixed)
        self.write_report("live-extent", 1, document=live)
        pairs = analyzer.discover_pairs(
            self.root, analyzer.CONTRACTS["rate-limited"])
        self.assertEqual(pairs[0][1].metrics["recall_after"], 0.931)

        self.temporary.cleanup()
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        live["recall"]["recall"] = 0.939
        self.write_report("fixed", 1, document=fixed)
        self.write_report("live-extent", 1, document=live)
        with self.assertRaisesRegex(
                analyzer.ReportError, "initial_recall"):
            analyzer.discover_pairs(
                self.root, analyzer.CONTRACTS["rate-limited"])


if __name__ == "__main__":
    unittest.main()
