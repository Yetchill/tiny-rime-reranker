#pragma once

#include <chrono>
#include <string>
#include <vector>

#include "backend.h"

namespace tinyrime {

struct GateConfig {
  float alpha = 0.25F;
  float confidence_threshold = 0.80F;
  float margin_threshold = 0.15F;
  std::chrono::microseconds deadline = std::chrono::milliseconds(3);
  bool require_context = true;
};

struct Decision {
  std::vector<size_t> order;
  bool changed = false;
  std::string reason;
};

class ConservativeReranker {
 public:
  ConservativeReranker(Backend* backend, GateConfig config = {});
  Decision Rerank(const ScoreRequest& request) const noexcept;

 private:
  Backend* backend_;
  GateConfig config_;
};

}  // namespace tinyrime
