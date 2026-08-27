#include <rime_api.h>
#include <rime/commit_history.h>
#include <rime/context.h>
#include <rime/service.h>

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>

namespace {

std::string Escape(const char* value) {
  if (value == nullptr) return "";
  std::ostringstream output;
  for (const unsigned char character : std::string(value)) {
    switch (character) {
      case '\\': output << "\\\\"; break;
      case '"': output << "\\\""; break;
      case '\n': output << "\\n"; break;
      case '\r': output << "\\r"; break;
      case '\t': output << "\\t"; break;
      default:
        if (character < 0x20) {
          output << "?";
        } else {
          output << character;
        }
    }
  }
  return output.str();
}

std::string JsonString(const std::string& line, const std::string& key) {
  const std::string marker = "\"" + key + "\"";
  size_t cursor = line.find(marker);
  if (cursor == std::string::npos) return "";
  cursor = line.find(':', cursor + marker.size());
  if (cursor == std::string::npos) return "";
  cursor = line.find('"', cursor + 1);
  if (cursor == std::string::npos) return "";
  ++cursor;
  std::string result;
  bool escaped = false;
  for (; cursor < line.size(); ++cursor) {
    const char character = line[cursor];
    if (escaped) {
      switch (character) {
        case 'n': result.push_back('\n'); break;
        case 'r': result.push_back('\r'); break;
        case 't': result.push_back('\t'); break;
        default: result.push_back(character); break;
      }
      escaped = false;
    } else if (character == '\\') {
      escaped = true;
    } else if (character == '"') {
      return result;
    } else {
      result.push_back(character);
    }
  }
  return "";
}

const char* Argument(int argc, char** argv, const std::string& name, const char* fallback = nullptr) {
  for (int index = 1; index + 1 < argc; ++index) {
    if (argv[index] == name) return argv[index + 1];
  }
  return fallback;
}

bool HasFlag(int argc, char** argv, const std::string& name) {
  for (int index = 1; index < argc; ++index) {
    if (argv[index] == name) return true;
  }
  return false;
}

void PrintError(const std::string& error) {
  std::cout << "{\"error\":\"" << Escape(error.c_str()) << "\",\"candidates\":[]}" << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  const char* shared_data = Argument(argc, argv, "--shared-data");
  const char* user_data = Argument(argc, argv, "--user-data");
  const char* prebuilt_data = Argument(argc, argv, "--prebuilt-data", user_data);
  const char* schema = Argument(argc, argv, "--schema", "rime_ice");
  const int top_k = std::atoi(Argument(argc, argv, "--top-k", "8"));
  if (shared_data == nullptr || user_data == nullptr || top_k < 1 || top_k > 64) {
    std::cerr << "usage: tinyrime_rime_runner --shared-data DIR --user-data DIR "
                 "[--schema ID] [--top-k N]\n";
    return 2;
  }

  RimeApi* rime = rime_get_api();
  RIME_STRUCT(RimeTraits, traits);
  traits.shared_data_dir = shared_data;
  traits.user_data_dir = user_data;
  traits.prebuilt_data_dir = prebuilt_data;
  traits.staging_dir = user_data;
  traits.app_name = "tinyrime.runner";
  traits.min_log_level = 2;
  traits.log_dir = "";
  rime->setup(&traits);
  rime->initialize(&traits);
  if (!HasFlag(argc, argv, "--skip-maintenance") && rime->start_maintenance(True)) {
    rime->join_maintenance_thread();
  }
  const std::string schema_file = std::string(shared_data) + "/" + schema + ".schema.yaml";
  if (!HasFlag(argc, argv, "--skip-deploy") && !rime->deploy_schema(schema_file.c_str())) {
    std::cerr << "failed to deploy schema: " << schema_file << "\n";
    rime->finalize();
    return 3;
  }
  const RimeSessionId session = rime->create_session();
  if (!session || !rime->select_schema(session, schema)) {
    std::cerr << "failed to create session or select schema: " << schema << "\n";
    if (session) rime->destroy_session(session);
    rime->finalize();
    return 3;
  }

  std::string line;
  while (std::getline(std::cin, line)) {
    const std::string pinyin = JsonString(line, "pinyin");
    const std::string context_text = JsonString(line, "context");
    if (pinyin.empty()) {
      PrintError("pinyin is required");
      continue;
    }
    rime->clear_composition(session);
    if (auto internal_session = rime::Service::instance().GetSession(session)) {
      if (auto* internal_context = internal_session->context()) {
        internal_context->commit_history().clear();
        if (!context_text.empty()) {
          internal_context->commit_history().Push(rime::CommitRecord{"tinyrime_fixture", context_text});
        }
      }
    }
    if (!rime->simulate_key_sequence(session, pinyin.c_str())) {
      PrintError("Rime rejected key sequence");
      continue;
    }
    RIME_STRUCT(RimeContext, context);
    if (!rime->get_context(session, &context)) {
      PrintError("Rime context unavailable");
      continue;
    }
    std::cout << "{\"pinyin\":\"" << Escape(pinyin.c_str()) << "\",\"context\":\""
              << Escape(context_text.c_str()) << "\",\"candidates\":[";
    RimeCandidateListIterator iterator;
    int rank = 0;
    bool first = true;
    if (rime->candidate_list_begin(session, &iterator)) {
      while (rank < top_k && rime->candidate_list_next(&iterator)) {
        if (!first) std::cout << ',';
        first = false;
        std::cout << "{\"text\":\"" << Escape(iterator.candidate.text) << "\",\"rank\":"
                  << rank << ",\"quality\":null,\"type\":null,\"comment\":\""
                  << Escape(iterator.candidate.comment) << "\",\"preedit\":\""
                  << Escape(context.composition.preedit) << "\"}";
        ++rank;
      }
      rime->candidate_list_end(&iterator);
    }
    std::cout << "]}" << std::endl;
    rime->free_context(&context);
  }

  rime->destroy_session(session);
  rime->finalize();
  return 0;
}
