#include <rime_api.h>
#include <rime/common.h>
#include <rime/registry.h>

#include "tinyrime/filter.h"

static void rime_tinyrime_initialize() {
  rime::Registry::instance().Register(
      "tinyrime_filter", new rime::Component<tinyrime::TinyrimeFilter>);
}

static void rime_tinyrime_finalize() { tinyrime::BackendRegistry::Set(nullptr); }

RIME_REGISTER_MODULE(tinyrime)
