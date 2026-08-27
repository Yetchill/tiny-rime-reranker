#include "tinyrime/reranker.h"

#include <cassert>
#include <chrono>
#include <thread>

using tinyrime::CandidateFeatures;
using tinyrime::ConservativeReranker;
using tinyrime::GateConfig;
using tinyrime::MockBackend;
using tinyrime::ScoreRequest;
using tinyrime::ScoreResult;

int main() {
  ScoreRequest request{"项目正在", {"shi", "shi"}, {{"事实", 0}, {"实施", 1}, {"试试", 2}}};
  MockBackend promoting({{0.0F, 8.0F, 0.0F}, 0.99F});
  auto changed = ConservativeReranker(&promoting).Rerank(request);
  assert(changed.changed);
  assert(changed.order.front() == 1);

  MockBackend uncertain({{0.0F, 8.0F, 0.0F}, 0.20F});
  auto abstained = ConservativeReranker(&uncertain).Rerank(request);
  assert(!abstained.changed);
  assert(abstained.order.front() == 0);

  MockBackend failed({}, false);
  auto fallback = ConservativeReranker(&failed).Rerank(request);
  assert(!fallback.changed);
  assert(fallback.order.front() == 0);
  return 0;
}
