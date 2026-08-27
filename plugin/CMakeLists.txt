cmake_minimum_required(VERSION 3.16)
project(tinyrime_plugin LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(tinyrime_runtime STATIC
  src/reranker.cc
  ../runtime/native/native_mlp_backend.cc
)
target_include_directories(tinyrime_runtime PUBLIC include ../runtime/interface ../runtime/native)
target_compile_options(tinyrime_runtime PRIVATE -Wall -Wextra -Wpedantic)

enable_testing()
add_executable(tinyrime_runtime_test tests/reranker_test.cc)
target_link_libraries(tinyrime_runtime_test PRIVATE tinyrime_runtime)
add_test(NAME tinyrime_runtime_test COMMAND tinyrime_runtime_test)

option(TINYRIME_BUILD_RIME_RUNNER "Build the headless runner against an installed librime" OFF)
if(TINYRIME_BUILD_RIME_RUNNER)
  find_path(RIME_INCLUDE_DIR rime_api.h REQUIRED)
  find_library(RIME_LIBRARY rime REQUIRED)
  add_executable(tinyrime_rime_runner src/rime_runner.cc)
  target_include_directories(tinyrime_rime_runner PRIVATE ${RIME_INCLUDE_DIR})
  target_link_libraries(tinyrime_rime_runner PRIVATE ${RIME_LIBRARY})
endif()
