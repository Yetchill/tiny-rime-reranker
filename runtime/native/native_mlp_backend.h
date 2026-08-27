#pragma once

#include <cstddef>
#include <vector>

#include "backend.h"

namespace tinyrime {

class NativeMLPBackend final : public Backend {
 public:
  struct Weights {
    size_t input_size = 0;
    size_t hidden_size = 0;
    std::vector<float> input;
    std::vector<float> input_bias;
    std::vector<float> output;
    float output_bias = 0.0F;
    std::vector<float> gate;
    float gate_bias = 0.0F;
  };

  explicit NativeMLPBackend(Weights weights);
  bool Score(const ScoreRequest& request, ScoreResult* result) noexcept override;

 private:
  std::vector<float> Features(const ScoreRequest& request, size_t candidate_index) const;
  Weights weights_;
};

}  // namespace tinyrime
