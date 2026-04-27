import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple


def ns_to_ms(ns: float) -> float:
    return ns / 1e6


def fmt_pct(x: float) -> str:
    return f"{x:.2f}%"


def fmt_ms_from_ns(ns: float) -> str:
    return f"{ns_to_ms(ns):.3f} ms"


def fmt_value(x: Any) -> str:
    if isinstance(x, int):
        return f"{x:,}"
    if isinstance(x, float):
        return f"{x:,.6f}".rstrip("0").rstrip(".")
    return str(x)


def is_number(x: Any) -> bool:
    return isinstance(x, (int, float))


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def sort_dict_desc(d: Dict[str, Any], drop_zero: bool = False) -> List[Tuple[str, Any]]:
    items = []
    for k, v in d.items():
        if not is_number(v):
            continue
        if drop_zero and v == 0:
            continue
        items.append((k, v))
    return sorted(items, key=lambda kv: kv[1], reverse=True)


def markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def dict_to_bullets(d: Dict[str, Any], indent: int = 0) -> List[str]:
    lines = []
    prefix = "  " * indent
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append(f"{prefix}- **{k}**")
            lines.extend(dict_to_bullets(v, indent + 1))
        else:
            lines.append(f"{prefix}- **{k}**: {fmt_value(v)}")
    return lines


def build_percentage_table(
    d: Dict[str, Any],
    total: float,
    drop_zero: bool = True,
) -> str:
    items = sort_dict_desc(d, drop_zero=drop_zero)
    rows = []
    for k, v in items:
        pct = safe_div(v, total) * 100
        rows.append([k, fmt_ms_from_ns(v), fmt_pct(pct)])
    if not rows:
        return "无有效数据"
    return markdown_table(["部分", "时间", "占比"], rows)


def build_generic_table(d: Dict[str, Any], value_title: str = "值", sort_numeric: bool = False) -> str:
    rows = []
    if sort_numeric:
        numeric_items = sort_dict_desc(d, drop_zero=False)
        used = set()
        for k, v in numeric_items:
            rows.append([k, fmt_value(v)])
            used.add(k)
        for k, v in d.items():
            if k in used:
                continue
            rows.append([k, fmt_value(v)])
    else:
        for k, v in d.items():
            rows.append([k, fmt_value(v)])

    if not rows:
        return "无"
    return markdown_table(["字段", value_title], rows)


def top_items_text(title: str, d: Dict[str, Any], total: float, topn: int = 3, drop_zero: bool = True) -> str:
    items = sort_dict_desc(d, drop_zero=drop_zero)[:topn]
    if not items:
        return f"- {title}：无有效数据。"

    parts = []
    for k, v in items:
        pct = safe_div(v, total) * 100
        parts.append(f"`{k}`（{fmt_pct(pct)}）")
    return f"- {title}：占比最高的几项是 " + "、".join(parts) + "。"


def build_latency_section(latency: Dict[str, Any]) -> str:
    rows = []
    for k, v in latency.items():
        if is_number(v) and k.endswith("_ns"):
            rows.append([k, fmt_ms_from_ns(v)])
        else:
            rows.append([k, fmt_value(v)])
    if not rows:
        return "无"
    return markdown_table(["延迟字段", "值"], rows)


def build_sub_breakdown(sub_breakdown: Dict[str, Any], drop_zero: bool = True) -> List[str]:
    lines = []
    if not isinstance(sub_breakdown, dict) or not sub_breakdown:
        lines.append("无")
        return lines

    for group_name, group_dict in sub_breakdown.items():
        if not isinstance(group_dict, dict):
            continue

        numeric_items = [(k, v) for k, v in group_dict.items() if is_number(v)]
        total = sum(v for _, v in numeric_items)

        lines.append(f"#### {group_name}")
        lines.append("")
        if total > 0:
            lines.append(build_percentage_table(group_dict, total, drop_zero=drop_zero))
            lines.append("")
            lines.append(top_items_text(f"{group_name} 内部热点", group_dict, total, drop_zero=drop_zero))
        else:
            lines.append("无有效数据")
        lines.append("")

    return lines


def find_operation_blocks(data: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    result = []
    for k, v in data.items():
        if k.endswith("_breakdown") and isinstance(v, dict) and "operation" in v:
            result.append((k, v))
    return result


def build_operation_section(op_name: str, op_data: Dict[str, Any], drop_zero: bool = True) -> List[str]:
    lines = []
    breakdown = op_data.get("breakdown", {})
    sub_breakdown = op_data.get("sub_breakdown", {})
    latency = op_data.get("latency", {})
    counters = op_data.get("counters", {})
    count = op_data.get("count", 0)

    total_breakdown = sum(v for v in breakdown.values() if is_number(v))

    lines.append(f"## {op_name.upper()} 分析")
    lines.append("")
    lines.append(f"- 操作数：**{fmt_value(count)}**")
    if "mean_end_to_end_ns" in latency:
        lines.append(f"- 平均端到端延迟：**{fmt_ms_from_ns(latency['mean_end_to_end_ns'])}**")
    if "p50_end_to_end_ns" in latency:
        lines.append(f"- P50 端到端延迟：**{fmt_ms_from_ns(latency['p50_end_to_end_ns'])}**")
    if "p95_end_to_end_ns" in latency:
        lines.append(f"- P95 端到端延迟：**{fmt_ms_from_ns(latency['p95_end_to_end_ns'])}**")
    if "p99_end_to_end_ns" in latency:
        lines.append(f"- P99 端到端延迟：**{fmt_ms_from_ns(latency['p99_end_to_end_ns'])}**")
    lines.append("")

    lines.append("### 一级 Breakdown 占比")
    lines.append("")
    if total_breakdown > 0:
        lines.append(build_percentage_table(breakdown, total_breakdown, drop_zero=drop_zero))
        lines.append("")
        lines.append(top_items_text(f"{op_name} 一级热点", breakdown, total_breakdown, drop_zero=drop_zero))
    else:
        lines.append("无有效数据")
    lines.append("")

    lines.append("### Sub Breakdown 细分占比")
    lines.append("")
    lines.extend(build_sub_breakdown(sub_breakdown, drop_zero=drop_zero))

    lines.append("### Latency")
    lines.append("")
    lines.append(build_latency_section(latency))
    lines.append("")

    lines.append("### Counters")
    lines.append("")
    lines.append(build_generic_table(counters, value_title="值", sort_numeric=True))
    lines.append("")

    return lines


def build_compare_section(data: Dict[str, Any]) -> List[str]:
    lines = []
    insert_data = data.get("insert_breakdown", {})
    query_data = data.get("query_breakdown", {})

    insert_bd = insert_data.get("breakdown", {})
    query_bd = query_data.get("breakdown", {})

    if not insert_bd or not query_bd:
        return lines

    insert_total = sum(v for v in insert_bd.values() if is_number(v))
    query_total = sum(v for v in query_bd.values() if is_number(v))

    keys = sorted(set(insert_bd.keys()) | set(query_bd.keys()))
    rows = []
    for k in keys:
        ins = insert_bd.get(k, 0)
        qry = query_bd.get(k, 0)
        rows.append([
            k,
            fmt_pct(safe_div(ins, insert_total) * 100),
            fmt_pct(safe_div(qry, query_total) * 100),
        ])

    lines.append("## Insert / Query 对比")
    lines.append("")
    lines.append(markdown_table(["类别", "Insert 占比", "Query 占比"], rows))
    lines.append("")

    ins_top = sort_dict_desc(insert_bd, drop_zero=True)
    qry_top = sort_dict_desc(query_bd, drop_zero=True)

    if ins_top:
        k, v = ins_top[0]
        lines.append(f"- Insert 最大部分是 **{k}**，占 **{fmt_pct(safe_div(v, insert_total) * 100)}**。")
    if qry_top:
        k, v = qry_top[0]
        lines.append(f"- Query 最大部分是 **{k}**，占 **{fmt_pct(safe_div(v, query_total) * 100)}**。")

    if insert_total > 0 and query_total > 0:
        if safe_div(insert_bd.get("gpu_ns", 0), insert_total) > safe_div(query_bd.get("gpu_ns", 0), query_total):
            lines.append("- Insert 更偏向 GPU 计算密集。")
        else:
            lines.append("- Query 的 GPU 压力不低于 Insert。")

        if safe_div(query_bd.get("rdma_ns", 0), query_total) > safe_div(insert_bd.get("rdma_ns", 0), insert_total):
            lines.append("- Query 更偏向 RDMA / 远端访问受限。")
        else:
            lines.append("- Insert 的 RDMA 成本同样非常显著。")

    lines.append("")
    return lines


def build_report(data: Dict[str, Any], drop_zero: bool = True) -> str:
    lines = []
    lines.append("# Breakdown 分析报告")
    lines.append("")

    meta = data.get("meta")
    if isinstance(meta, dict):
        lines.append("## 实验元信息")
        lines.append("")
        lines.extend(dict_to_bullets(meta))
        lines.append("")

    bottleneck_summary = data.get("bottleneck_summary")
    if isinstance(bottleneck_summary, dict):
        lines.append("## Bottleneck Summary")
        lines.append("")
        for k, v in bottleneck_summary.items():
            lines.append(f"### {k}")
            lines.append("")
            lines.append("```text")
            lines.append(str(v).rstrip())
            lines.append("```")
            lines.append("")

    for block_key, block_value in find_operation_blocks(data):
        op_name = block_value.get("operation", block_key.replace("_breakdown", ""))
        lines.extend(build_operation_section(op_name, block_value, drop_zero=drop_zero))

    system_counters = data.get("system_counters")
    if isinstance(system_counters, dict):
        lines.append("## System Counters")
        lines.append("")
        lines.append(build_generic_table(system_counters, value_title="值", sort_numeric=True))
        lines.append("")

    lines.extend(build_compare_section(data))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="将 breakdown JSON 生成 Markdown 报告")
    parser.add_argument("input", help="输入 JSON 文件路径")
    parser.add_argument("-o", "--output", default="report.md", help="输出 Markdown 文件路径")
    parser.add_argument(
        "--keep-zero",
        action="store_true",
        help="保留值为 0 的 breakdown / sub_breakdown 项",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    report = build_report(data, drop_zero=not args.keep_zero)

    with output_path.open("w", encoding="utf-8") as f:
        f.write(report)

    print(f"报告已生成：{output_path.resolve()}")


if __name__ == "__main__":
    main()