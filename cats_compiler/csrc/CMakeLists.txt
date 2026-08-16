cmake_minimum_required(VERSION 3.20)
project(cats_compiler CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(LLVM REQUIRED CONFIG
  PATHS "${CMAKE_SOURCE_DIR}/../../third_party/llvm-project/llvm/build"
  NO_DEFAULT_PATH)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION} at ${LLVM_DIR}")

add_executable(cats_compiler main.cpp)

target_include_directories(cats_compiler PRIVATE ${LLVM_INCLUDE_DIRS})
target_compile_definitions(cats_compiler PRIVATE ${LLVM_DEFINITIONS})

# RTTI must match the LLVM build (this build is RTTI-OFF).
target_compile_options(cats_compiler PRIVATE -fno-rtti)

llvm_map_components_to_libnames(LLVM_LIBS core support)
target_link_libraries(cats_compiler PRIVATE ${LLVM_LIBS})

# ── Google Benchmark (pre-built inside the LLVM third-party tree) ─────────────
set(GBENCH_INCLUDE "${CMAKE_SOURCE_DIR}/../../third_party/llvm-project/third-party/benchmark/include")
set(GBENCH_LIB_DIR "${CMAKE_SOURCE_DIR}/../../third_party/llvm-project/llvm/build/lib")

function(add_model_benchmark TARGET_NAME MAIN_SRC SHIM_SRC)
    add_executable(${TARGET_NAME}
        bench/${MAIN_SRC}
        bench/${SHIM_SRC}
    )
    target_include_directories(${TARGET_NAME} PRIVATE ${GBENCH_INCLUDE})
    target_compile_options(${TARGET_NAME} PRIVATE -O3)
    target_link_libraries(${TARGET_NAME} PRIVATE
        ${GBENCH_LIB_DIR}/libbenchmark.a
        pthread
    )
endfunction()

add_model_benchmark(bench_classifier     bench_main.cpp     cb_classifier_shim.cpp)
add_model_benchmark(bench_constexpr      bench_main.cpp     cb_constexpr_shim.cpp)
add_model_benchmark(bench_big_classifier bench_big_main.cpp cb_big_cls_shim.cpp)
add_model_benchmark(bench_big_constexpr  bench_big_main.cpp cb_big_cls_constexpr_shim.cpp)
add_model_benchmark(bench_ctr            bench_ctr_main.cpp cb_ctr_shim.cpp)
