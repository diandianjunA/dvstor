#include "tools/breakdown_benchmark/args.hh"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace tools::breakdown_benchmark {
namespace {

template <typename T>
T parse_unsigned_argument(const std::string& text, const char* option) {
  static_assert(std::is_unsigned_v<T>);
  if (text.empty() || text.front() == static_cast<char>(45)) {
    throw std::runtime_error(std::string(option) + " must be a non-negative integer");
  }
  size_t parsed = 0;
  unsigned long long value = 0;
  try {
    value = std::stoull(text, &parsed, 10);
  } catch (const std::exception&) {
    throw std::runtime_error(std::string(option) + " has an invalid integer value: " + text);
  }
  if (parsed != text.size() || value > std::numeric_limits<T>::max()) {
    throw std::runtime_error(std::string(option) + " is outside its supported integer range: " + text);
  }
  return static_cast<T>(value);
}

double parse_double_argument(const std::string& text, const char* option) {
  size_t parsed = 0;
  double value = 0.0;
  try {
    value = std::stod(text, &parsed);
  } catch (const std::exception&) {
    throw std::runtime_error(std::string(option) + " has an invalid numeric value: " + text);
  }
  if (parsed != text.size() || !std::isfinite(value)) {
    throw std::runtime_error(std::string(option) + " must be a finite number: " + text);
  }
  return value;
}

}  // namespace


std::string trim(std::string value) {
  auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
  value.erase(value.begin(), std::find_if(value.begin(), value.end(), not_space));
  value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(), value.end());
  return value;
}

ConfigMap read_config(const std::string& path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open config: " + path);
  }

  ConfigMap config;
  std::string line;
  size_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    const auto comment_pos = line.find_first_of("#;");
    if (comment_pos != std::string::npos) {
      line.erase(comment_pos);
    }
    line = trim(line);
    if (line.empty()) {
      continue;
    }

    const auto eq_pos = line.find('=');
    if (eq_pos == std::string::npos) {
      throw std::runtime_error(
        "malformed config line " + std::to_string(line_number) +
        " (expected key=value): " + path);
    }
    auto key = trim(line.substr(0, eq_pos));
    auto value = trim(line.substr(eq_pos + 1));
    if (key.empty()) {
      throw std::runtime_error(
        "empty config key on line " + std::to_string(line_number) +
        ": " + path);
    }
    const auto [position, inserted] =
      config.emplace(std::move(key), std::move(value));
    if (!inserted) {
      throw std::runtime_error(
        "duplicate config key on line " + std::to_string(line_number) +
        ": " + position->first);
    }
  }
  return config;
}

bool is_truthy(const std::string& value) {
  return value == "1" || value == "true" || value == "on" || value == "yes";
}

bool is_falsey(const std::string& value) {
  return value == "0" || value == "false" || value == "off" || value == "no";
}

std::vector<std::string> split_tokens(const std::string& value) {
  std::string normalized = value;
  std::replace(normalized.begin(), normalized.end(), ',', ' ');
  std::stringstream ss(normalized);
  std::vector<std::string> tokens;
  std::string token;
  while (ss >> token) {
    tokens.push_back(token);
  }
  return tokens;
}

std::vector<char*> make_argv(std::vector<std::string>& args) {
  std::vector<char*> argv;
  argv.reserve(args.size());
  for (auto& arg : args) {
    argv.push_back(arg.data());
  }
  return argv;
}

std::vector<std::string> build_service_argv(const std::string& service_config_path) {
  const auto config = read_config(service_config_path);
  std::vector<std::string> args;
  args.emplace_back("dvstor_breakdown_benchmark");

  static const std::vector<std::string> multi_keys = {"servers", "clients", "storage-peers"};
  static const std::vector<std::string> flag_keys = {
    "initiator", "disable-thread-pinning"};
  static const std::vector<std::string> benchmark_only_keys = {"insert-start-id", "write-id-base"};

  for (const auto& [key, value] : config) {
    if (std::find(benchmark_only_keys.begin(), benchmark_only_keys.end(), key) != benchmark_only_keys.end()) {
      continue;
    }

    const std::string option = "--" + key;
    if (std::find(flag_keys.begin(), flag_keys.end(), key) != flag_keys.end()) {
      if (is_truthy(value)) {
        args.push_back(option);
      } else if (!is_falsey(value)) {
        throw std::runtime_error(
          "invalid boolean value for " + option + ": " + value);
      }
      continue;
    }

    if (std::find(multi_keys.begin(), multi_keys.end(), key) != multi_keys.end()) {
      const auto tokens = split_tokens(value);
      if (!tokens.empty()) {
        args.push_back(option);
        args.insert(args.end(), tokens.begin(), tokens.end());
      }
      continue;
    }

    args.push_back(option);
    args.push_back(value);
  }

  return args;
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    auto require_value = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + name);
      }
      return argv[++i];
    };

    if (flag == "--service-config") {
      args.service_config_path = require_value("--service-config");
    } else if (flag == "--profile-name") {
      args.profile_name = require_value("--profile-name");
    } else if (flag == "--system-variant-label") {
      args.system_variant_label = require_value("--system-variant-label");
    } else if (flag == "--workload") {
      args.workload = require_value("--workload");
    } else if (flag == "--warmup-ops") {
      args.warmup_ops = parse_unsigned_argument<size_t>(require_value("--warmup-ops"), "--warmup-ops");
    } else if (flag == "--measure-ops") {
      args.measure_ops = parse_unsigned_argument<size_t>(require_value("--measure-ops"), "--measure-ops");
    } else if (flag == "--warmup-seconds") {
      args.warmup_seconds = parse_unsigned_argument<size_t>(require_value("--warmup-seconds"), "--warmup-seconds");
    } else if (flag == "--measure-seconds") {
      args.measure_seconds = parse_unsigned_argument<size_t>(require_value("--measure-seconds"), "--measure-seconds");
    } else if (flag == "--client-threads") {
      args.client_threads = parse_unsigned_argument<size_t>(require_value("--client-threads"), "--client-threads");
    } else if (flag == "--read-ratio") {
      args.read_ratio = parse_double_argument(require_value("--read-ratio"), "--read-ratio");
    } else if (flag == "--mixed-mode") {
      args.mixed_mode = require_value("--mixed-mode");
    } else if (flag == "--write-threads") {
      args.write_threads = parse_unsigned_argument<size_t>(require_value("--write-threads"), "--write-threads");
    } else if (flag == "--target-query-qps") {
      args.target_query_qps = parse_double_argument(require_value("--target-query-qps"), "--target-query-qps");
    } else if (flag == "--target-write-qps" || flag == "--target-insert-qps") {
      args.target_write_qps = parse_double_argument(require_value(flag.c_str()), flag.c_str());
    } else if (flag == "--recall-query-file") {
      if (!args.recall_query_file.empty()) {
        throw std::runtime_error("--recall-query-file was specified more than once");
      }
      args.recall_query_file = require_value("--recall-query-file");
    } else if (flag == "--performance-query-file") {
      if (!args.performance_query_file.empty()) {
        throw std::runtime_error("--performance-query-file was specified more than once");
      }
      args.performance_query_file = require_value("--performance-query-file");
    } else if (flag == "--query-file") {
      if (!args.recall_query_file.empty()) {
        throw std::runtime_error("--query-file and --recall-query-file cannot both be specified");
      }
      args.recall_query_file = require_value("--query-file");
    } else if (flag == "--insert-file") {
      args.insert_file = require_value("--insert-file");
    } else if (flag == "--groundtruth-file") {
      args.groundtruth_file = require_value("--groundtruth-file");
    } else if (flag == "--recall-queries") {
      args.recall_queries = parse_unsigned_argument<size_t>(require_value("--recall-queries"), "--recall-queries");
    } else if (flag == "--recall-k") {
      args.recall_k = parse_unsigned_argument<uint32_t>(require_value("--recall-k"), "--recall-k");
    } else if (flag == "--recall-mode") {
      args.recall_mode = require_value("--recall-mode");
    } else if (flag == "--recall-base-id-limit") {
      args.recall_base_id_limit = parse_unsigned_argument<uint32_t>(
        require_value("--recall-base-id-limit"),
        "--recall-base-id-limit");
    } else if (flag == "--storage-maintenance-log") {
      args.storage_maintenance_logs.push_back(
        require_value("--storage-maintenance-log"));
    } else if (flag == "--recall-only") {
      args.recall_only = true;
    } else if (flag == "--synthetic") {
      args.synthetic = true;
    } else if (flag == "--report-json") {
      args.report_json_path = require_value("--report-json");
    } else if (flag == "--report-text") {
      args.report_text_path = require_value("--report-text");
    } else if (flag == "--insert-start-id") {
      args.insert_start_id = parse_unsigned_argument<uint32_t>(require_value("--insert-start-id"), "--insert-start-id");
    } else if (flag == "--write-insert-ratio") {
      args.write_insert_ratio = parse_double_argument(require_value("--write-insert-ratio"), "--write-insert-ratio");
    } else if (flag == "--write-upsert-ratio") {
      args.write_upsert_ratio = parse_double_argument(require_value("--write-upsert-ratio"), "--write-upsert-ratio");
    } else if (flag == "--write-delete-ratio") {
      args.write_delete_ratio = parse_double_argument(require_value("--write-delete-ratio"), "--write-delete-ratio");
    } else {
      throw std::runtime_error("unknown argument: " + flag);
    }
  }

  if (args.service_config_path.empty()) {
    throw std::runtime_error("--service-config is required");
  }
  if (args.report_json_path.empty()) {
    throw std::runtime_error("--report-json is required");
  }
  if (args.profile_name.empty() || args.system_variant_label.empty()) {
    throw std::runtime_error(
      "--profile-name and --system-variant-label must be non-empty");
  }
  if (args.workload != "query" && args.workload != "insert" && args.workload != "both" &&
      args.workload != "mixed") {
    throw std::runtime_error("--workload must be query, insert, both, or mixed");
  }
  if (args.client_threads == 0) {
    throw std::runtime_error("--client-threads must be > 0");
  }
  const std::vector<double> numeric_options = {
    args.read_ratio,
    args.target_query_qps,
    args.target_write_qps,
    args.write_insert_ratio,
    args.write_upsert_ratio,
    args.write_delete_ratio,
  };
  if (std::any_of(numeric_options.begin(), numeric_options.end(),
                  [](double value) { return !std::isfinite(value); })) {
    throw std::runtime_error("floating-point arguments must be finite");
  }
  if (args.read_ratio < 0.0 || args.read_ratio > 1.0) {
    throw std::runtime_error("--read-ratio must be in [0, 1]");
  }
  if (args.mixed_mode != "probability" && args.mixed_mode != "fixed_threads" &&
      args.mixed_mode != "rate_limited" &&
      args.mixed_mode != "write_rate_limited") {
    throw std::runtime_error(
      "--mixed-mode must be probability, fixed_threads, rate_limited, or "
      "write_rate_limited");
  }
  if (args.write_insert_ratio < 0.0 || args.write_upsert_ratio < 0.0 || args.write_delete_ratio < 0.0) {
    throw std::runtime_error("write mutation ratios must be >= 0");
  }
  if (args.write_insert_ratio + args.write_upsert_ratio + args.write_delete_ratio <= 0.0) {
    throw std::runtime_error("at least one write mutation ratio must be > 0");
  }
  if (args.target_query_qps < 0.0 || args.target_write_qps < 0.0) {
    throw std::runtime_error("target rates must be non-negative");
  }
  if (!args.groundtruth_file.empty() && args.recall_query_file.empty()) {
    throw std::runtime_error("--recall-query-file is required with --groundtruth-file");
  }
  if (args.recall_only && args.groundtruth_file.empty()) {
    throw std::runtime_error("--recall-only requires --groundtruth-file");
  }
  if (args.recall_mode != "all" && args.recall_mode != "base_only") {
    throw std::runtime_error("--recall-mode must be all or base_only");
  }
  if ((args.recall_mode == "base_only") !=
      (args.recall_base_id_limit != 0)) {
    throw std::runtime_error(
      "--recall-mode base_only and a positive --recall-base-id-limit "
      "must be supplied together");
  }
  if (args.recall_mode == "base_only" && args.groundtruth_file.empty()) {
    throw std::runtime_error(
      "--recall-mode base_only requires --groundtruth-file");
  }
  const bool workload_has_queries =
    !args.recall_only &&
    (args.workload == "query" || args.workload == "both" ||
     (args.workload == "mixed" &&
      (args.mixed_mode == "rate_limited" ? args.target_query_qps > 0.0
       : args.mixed_mode == "write_rate_limited" ? true
       : args.read_ratio > 0.0)));
  if (workload_has_queries && args.performance_query_file.empty()) {
    throw std::runtime_error(
      "--performance-query-file is required for query performance phases; "
      "the recall query file is intentionally not reused");
  }
  if (workload_has_queries && !args.recall_query_file.empty()) {
    std::error_code error;
    const bool same_file = std::filesystem::equivalent(
      args.recall_query_file, args.performance_query_file, error);
    if (!error && same_file) {
      throw std::runtime_error(
        "--recall-query-file and --performance-query-file must refer to different files");
    }
  }
  if (!args.performance_query_file.empty() && !args.insert_file.empty()) {
    std::error_code error;
    const bool same_file = std::filesystem::equivalent(
      args.performance_query_file, args.insert_file, error);
    if (!error && same_file) {
      throw std::runtime_error(
        "--performance-query-file and --insert-file must refer to different files");
    }
  }
  if (args.insert_start_id == 0) {
    const auto service_config = read_config(args.service_config_path);
    auto it = service_config.find("insert-start-id");
    if (it == service_config.end()) {
      it = service_config.find("write-id-base");
    }
    if (it != service_config.end() && !it->second.empty()) {
      args.insert_start_id = parse_unsigned_argument<uint32_t>(it->second, "insert-start-id in service config");
    }
  }
  const bool use_time_mode = args.warmup_seconds > 0 || args.measure_seconds > 0;
  if (use_time_mode && (args.warmup_seconds == 0 || args.measure_seconds == 0)) {
    throw std::runtime_error("--warmup-seconds and --measure-seconds must both be > 0 when using time-based mode");
  }
  if (args.mixed_mode == "rate_limited") {
    if (args.workload != "mixed" || !use_time_mode) {
      throw std::runtime_error(
        "--mixed-mode rate_limited requires --workload mixed and time-based mode");
    }
    if (args.target_query_qps <= 0.0 && args.target_write_qps <= 0.0) {
      throw std::runtime_error(
        "rate_limited mode requires a positive query or write target");
    }
    if (args.write_threads != 0) {
      throw std::runtime_error(
        "--write-threads is only valid with write_rate_limited mode");
    }
  } else if (args.mixed_mode == "write_rate_limited") {
    if (args.workload != "mixed" || !use_time_mode) {
      throw std::runtime_error(
        "--mixed-mode write_rate_limited requires --workload mixed and "
        "time-based mode");
    }
    if (args.target_query_qps != 0.0 || args.target_write_qps <= 0.0) {
      throw std::runtime_error(
        "write_rate_limited requires zero target query QPS and a positive "
        "target write QPS");
    }
    if (args.write_threads == 0 || args.write_threads >= args.client_threads) {
      throw std::runtime_error(
        "write_rate_limited requires --write-threads in [1, client_threads)");
    }
  } else if (args.target_query_qps != 0.0 || args.target_write_qps != 0.0) {
    throw std::runtime_error(
      "target rates require rate_limited or write_rate_limited mode");
  } else if (args.write_threads != 0) {
    throw std::runtime_error(
      "--write-threads is only valid with write_rate_limited mode");
  }
  return args;
}


}  // namespace tools::breakdown_benchmark
