#include "native_mlp_backend.h"
#include "tinyrime/reranker.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

int main() {
  constexpr size_t kInput = 256;
  constexpr size_t kHidden = 64;
  tinyrime::NativeMLPBackend::Weights weights;
  weights.input_size = kInput;
  weights.hidden_size = kHidden;
  weights.input.resize(kInput * kHidden);
  weights.input_bias.resize(kHidden);
  weights.output.resize(kHidden);
  weights.gate.resize(kHidden);
  for (size_t index = 0; index < weights.input.size(); ++index) {
    weights.input[index] = static_cast<float>(static_cast<int>(index % 17) - 8) / 8192.0F;
  }
  for (size_t index = 0; index < kHidden; ++index) {
    weights.output[index] = static_cast<float>(index % 7) / 512.0F;
    weights.gate[index] = static_cast<float>(index % 5) / 512.0F;
  }
  weights.gate_bias = 2.0F;
  tinyrime::NativeMLPBackend backend(std::move(weights));
  tinyrime::ConservativeReranker reranker(&backend);
  tinyrime::ScoreRequest request{
      "研究人员正在分析这项技术的实际效果",
      {"shi", "shi"},
      {{"事实", 0, 1.2, 1}, {"实施", 1, 1.1, 1}, {"试试", 2, 1.0, 1},
       {"实时", 3, 0.9, 1}, {"适时", 4, 0.8, 1}, {"时时", 5, 0.7, 1},
       {"实事", 6, 0.6, 1}, {"时事", 7, 0.5, 1}}};
  for (int index = 0; index < 1000; ++index) reranker.Rerank(request);
  std::vector<double> microseconds;
  microseconds.reserve(10000);
  for (int index = 0; index < 10000; ++index) {
    const auto started = std::chrono::steady_clock::now();
    reranker.Rerank(request);
    const auto elapsed = std::chrono::steady_clock::now() - started;
    microseconds.push_back(std::chrono::duration<double, std::micro>(elapsed).count());
  }
  std::sort(microseconds.begin(), microseconds.end());
  const auto percentile = [&](double fraction) {
    return microseconds[static_cast<size_t>(fraction * static_cast<double>(microseconds.size() - 1))];
  };
  std::cout << std::fixed << std::setprecision(4)
            << "{\"backend\":\"synthetic-native-mlp-256x64\",\"iterations\":10000,"
            << "\"p50_ms\":" << percentile(0.50) / 1000.0 << ",\"p95_ms\":"
            << percentile(0.95) / 1000.0 << ",\"p99_ms\":" << percentile(0.99) / 1000.0
            << ",\"max_ms\":" << microseconds.back() / 1000.0
            << ",\"scope\":\"runtime-microbenchmark-not-trained-model\"}\n";
}
