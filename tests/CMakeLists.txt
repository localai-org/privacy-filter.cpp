# Tests needing model assets carry the "model" label and skip (exit 77) when
# PF_GGUF_DIR is unset. Fast gate: ctest -LE model.
function(pf_add_test name)
  add_executable(${name} ${name}.cpp)
  target_link_libraries(${name} PRIVATE pf)
  target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/src)
  target_compile_options(${name} PRIVATE -Wall -Wextra)
  add_test(NAME ${name} COMMAND ${name})
  set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77)
endfunction()

pf_add_test(test_loader)
pf_add_test(test_graph_blocks)
pf_add_test(test_ner)
pf_add_test(test_tokenizer)

# CLI serialization needs neither model weights nor ggml.
add_executable(test_json test_json.cpp)
target_compile_options(test_json PRIVATE -Wall -Wextra)
add_test(NAME test_json COMMAND test_json)

pf_add_test(test_parity)
pf_add_test(test_window_stitch)
set_tests_properties(test_parity test_window_stitch test_tokenizer
    PROPERTIES ENVIRONMENT "PF_FIXTURES=${CMAKE_SOURCE_DIR}/tests/fixtures/hf")
set_tests_properties(test_parity test_window_stitch PROPERTIES LABELS "model")
