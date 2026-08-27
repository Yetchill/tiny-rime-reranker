#include "native_mlp_backend.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace tinyrime {
namespace {
float Sigmoid(float value) { return 1.0F / (1.0F + std::exp(-value)); }

uint64_t Hash(const std::string& value) {
  uint64_t hash = 1469598103934665603ULL;
  for (unsigned char byte : value) {
    hash ^= byte;
    hash *= 1099511628211ULL;
  }
  return hash;
}
}  // namespace

NativeMLPBackend::NativeMLPBackend(Weights weights) : weights_(std::move(weights)) {}

std::vector<float> NativeMLPBackend::Features(const ScoreRequest& request,
                                              size_t candidate_index) const {
  std::vector<float> features(weights_.input_size, 0.0F);
  if (features.empty()) return features;
  const auto& candidate = request.candidates[candidate_index];
  features[Hash("ctx:" + request.left_context) % features.size()] += 1.0F;
  features[Hash("cand:" + candidate.text) % features.size()] += 1.0F;
  for (const auto& syllable : request.pinyin) {
    features[Hash("py:" + syllable) % features.size()] += 1.0F;
  }
  if (features.size() >= 4) {
    features[0] = static_cast<float>(candidate.rime_rank) / 7.0F;
    features[1] = static_cast<float>(candidate.rime_quality);
    features[2] = static_cast<float>(candidate.text.size()) / 24.0F;
    features[3] = static_cast<float>(candidate.type_id) / 8.0F;
  }
  return features;
}

bool NativeMLPBackend::Score(const ScoreRequest& request, ScoreResult* result) noexcept {
  if (result == nullptr || weights_.input_size == 0 || weights_.hidden_size == 0 ||
      weights_.input.size() != weights_.input_size * weights_.hidden_size ||
      weights_.input_bias.size() != weights_.hidden_size ||
      weights_.output.size() != weights_.hidden_size ||
      weights_.gate.size() != weights_.hidden_size) {
    return false;
  }
  result->residuals.clear();
  result->residuals.reserve(request.candidates.size());
  float gate_sum = 0.0F;
  for (size_t candidate_index = 0; candidate_index < request.candidates.size(); ++candidate_index) {
    const auto features = Features(request, candidate_index);
    std::vector<float> hidden(weights_.hidden_size);
    for (size_t output_index = 0; output_index < weights_.hidden_size; ++output_index) {
      float value = weights_.input_bias[output_index];
      for (size_t input_index = 0; input_index < weights_.input_size; ++input_index) {
        value += features[input_index] *
                 weights_.input[output_index * weights_.input_size + input_index];
      }
      hidden[output_index] = std::max(0.0F, value);
    }
    float residual = weights_.output_bias;
    float gate_logit = weights_.gate_bias;
    for (size_t index = 0; index < hidden.size(); ++index) {
      residual += hidden[index] * weights_.output[index];
      gate_logit += hidden[index] * weights_.gate[index];
    }
    result->residuals.push_back(residual);
    gate_sum += gate_logit;
  }
  result->confidence = request.candidates.empty()
                           ? 0.0F
                           : Sigmoid(gate_sum / static_cast<float>(request.candidates.size()));
  return true;
}

}  // namespace tinyrime
