#pragma once

#include <memory>

#include <rime/filter.h>

#include "backend.h"

namespace tinyrime {

class BackendRegistry {
 public:
  static std::shared_ptr<Backend> Get();
  static void Set(std::shared_ptr<Backend> backend);
};

class TinyrimeFilter final : public rime::Filter {
 public:
  explicit TinyrimeFilter(const rime::Ticket& ticket);
  rime::an<rime::Translation> Apply(rime::an<rime::Translation> translation,
                                    rime::CandidateList* candidates) override;
};

}  // namespace tinyrime
