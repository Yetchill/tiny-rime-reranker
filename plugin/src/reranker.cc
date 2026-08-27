#include "tinyrime/reranker.h"

#include <algorithm>
#include <numeric>

namespace tinyrime {

ConservativeReranker::ConservativeReranker(Backend* backend, GateConfig config)
    : backend_(backend), config_(config) {}

Decision ConservativeReranker::Rerank(const ScoreRequest& request) const noexcept {
  const auto started = std::chrono::steady_clock::now();
  Decision decision;
  decision.order.resize(request.candidates.size());
  std::iota(decision.order.begin(), decision.order.end(), 0);
  auto abstain = [&decision](const char* reason) {
    decision.changed = false;
    decision.reason = reason;
    return decision;
  };

  if (backend_ == nullptr || request.candidates.empty()) return abstain("no-backend-or-candidates");
  if (request.candidates.size() > 8) return abstain("too-many-candidates");
  if (config_.require_context && request.left_context.empty()) return abstain("no-context");

  ScoreResult scores;
  if (!backend_->Score(request, &scores)) return abstain("backend-failure");
  if (std::chrono::steady_clock::now() - started > config_.deadline) return abstain("deadline");
  if (scores.residuals.size() != request.candidates.size()) return abstain("invalid-score-count");
  if (scores.confidence < config_.confidence_threshold) return abstain("low-confidence");

  std::vector<float> final_scores(request.candidates.size());
  for (size_t index = 0; index < request.candidates.size(); ++index) {
    final_scores[index] = -static_cast<float>(request.candidates[index].rime_rank) +
                          config_.alpha * scores.residuals[index];
  }
  std::stable_sort(decision.order.begin(), decision.order.end(), [&](size_t left, size_t right) {
    return final_scores[left] > final_scores[right];
  });
  if (decision.order.front() == 0) return abstain("top1-unchanged");
  if (final_scores[decision.order.front()] - final_scores[0] < config_.margin_threshold) {
    std::iota(decision.order.begin(), decision.order.end(), 0);
    return abstain("low-margin");
  }
  if (std::chrono::steady_clock::now() - started > config_.deadline) {
    std::iota(decision.order.begin(), decision.order.end(), 0);
    return abstain("deadline");
  }
  decision.changed = true;
  decision.reason = "promoted";
  return decision;
}

}  // namespace tinyrime
