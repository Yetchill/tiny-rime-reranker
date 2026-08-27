#pragma once

#include <string>
#include <utility>
#include <vector>

namespace tinyrime {

struct CandidateFeatures {
  std::string text;
  int rime_rank = 0;
  double rime_quality = 0.0;
  int type_id = 0;
};

struct ScoreRequest {
  std::string left_context;
  std::vector<std::string> pinyin;
  std::vector<CandidateFeatures> candidates;
};

struct ScoreResult {
  std::vector<float> residuals;
  float confidence = 0.0F;
};

class Backend {
 public:
  virtual ~Backend() = default;
  virtual bool Score(const ScoreRequest& request, ScoreResult* result) noexcept = 0;
};

class MockBackend final : public Backend {
 public:
  explicit MockBackend(ScoreResult result, bool succeeds = true)
      : result_(std::move(result)), succeeds_(succeeds) {}

  bool Score(const ScoreRequest&, ScoreResult* result) noexcept override {
    if (!succeeds_ || result == nullptr) return false;
    *result = result_;
    return true;
  }

 private:
  ScoreResult result_;
  bool succeeds_;
};

}  // namespace tinyrime
