#include "tinyrime/filter.h"

#include <algorithm>
#include <deque>
#include <mutex>
#include <utility>

#include <rime/candidate.h>
#include <rime/commit_history.h>
#include <rime/context.h>
#include <rime/engine.h>
#include <rime/translation.h>

#include "tinyrime/reranker.h"

namespace tinyrime {
namespace {
std::mutex backend_mutex;
std::shared_ptr<Backend> registered_backend;

std::string LastUtf8Codepoints(const std::string& value, size_t maximum) {
  size_t cursor = value.size();
  size_t count = 0;
  while (cursor > 0 && count < maximum) {
    --cursor;
    while (cursor > 0 && (static_cast<unsigned char>(value[cursor]) & 0xC0U) == 0x80U) --cursor;
    ++count;
  }
  return value.substr(cursor);
}

std::string LeftContext(rime::Context* context) {
  if (context == nullptr) return {};
  std::deque<std::string> chunks;
  size_t bytes = 0;
  for (auto iterator = context->commit_history().rbegin();
       iterator != context->commit_history().rend() && bytes < 256; ++iterator) {
    if (iterator->type == "raw" || iterator->type == "thru" || iterator->text.empty()) continue;
    chunks.push_front(iterator->text);
    bytes += iterator->text.size();
  }
  std::string combined;
  for (const auto& chunk : chunks) combined += chunk;
  return LastUtf8Codepoints(combined, 32);
}

int TypeId(const std::string& type) {
  if (type == "table") return 1;
  if (type == "user_table") return 2;
  if (type == "sentence") return 3;
  if (type == "completion") return 4;
  return 0;
}

class RerankedTranslation final : public rime::Translation {
 public:
  RerankedTranslation(rime::an<rime::Translation> upstream,
                      rime::CandidateList buffer,
                      const ScoreRequest& request,
                      Backend* backend)
      : upstream_(std::move(upstream)), buffer_(std::move(buffer)) {
    ConservativeReranker reranker(backend);
    const auto decision = reranker.Rerank(request);
    if (decision.order.size() == buffer_.size()) {
      rime::CandidateList reordered;
      reordered.reserve(buffer_.size());
      for (size_t index : decision.order) reordered.push_back(buffer_[index]);
      buffer_ = std::move(reordered);
    }
    set_exhausted(buffer_.empty() && (!upstream_ || upstream_->exhausted()));
  }

  bool Next() override {
    if (exhausted()) return false;
    if (cursor_ < buffer_.size()) {
      ++cursor_;
    } else if (upstream_ && !upstream_->exhausted()) {
      upstream_->Next();
    }
    if (cursor_ >= buffer_.size() && (!upstream_ || upstream_->exhausted())) set_exhausted(true);
    return !exhausted();
  }

  rime::an<rime::Candidate> Peek() override {
    if (exhausted()) return nullptr;
    if (cursor_ < buffer_.size()) return buffer_[cursor_];
    return upstream_ && !upstream_->exhausted() ? upstream_->Peek() : nullptr;
  }

 private:
  rime::an<rime::Translation> upstream_;
  rime::CandidateList buffer_;
  size_t cursor_ = 0;
};
}  // namespace

std::shared_ptr<Backend> BackendRegistry::Get() {
  std::lock_guard<std::mutex> lock(backend_mutex);
  return registered_backend;
}

void BackendRegistry::Set(std::shared_ptr<Backend> backend) {
  std::lock_guard<std::mutex> lock(backend_mutex);
  registered_backend = std::move(backend);
}

TinyrimeFilter::TinyrimeFilter(const rime::Ticket& ticket) : Filter(ticket) {}

rime::an<rime::Translation> TinyrimeFilter::Apply(rime::an<rime::Translation> translation,
                                                  rime::CandidateList*) {
  auto backend = BackendRegistry::Get();
  if (!translation || !backend || !engine_ || !engine_->context()) return translation;
  ScoreRequest request;
  request.left_context = LeftContext(engine_->context());
  if (request.left_context.empty()) return translation;
  request.pinyin = {engine_->context()->input()};
  rime::CandidateList prefetched;
  while (prefetched.size() < 8 && !translation->exhausted()) {
    if (auto candidate = translation->Peek()) {
      prefetched.push_back(candidate);
      request.candidates.push_back(
          {candidate->text(), static_cast<int>(prefetched.size() - 1), candidate->quality(), TypeId(candidate->type())});
    }
    translation->Next();
  }
  return rime::New<RerankedTranslation>(translation, std::move(prefetched), request, backend.get());
}

}  // namespace tinyrime
