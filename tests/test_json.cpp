// SPDX-License-Identifier: MIT
#include "../tools/json.h"

#include <cstdio>
#include <string>
#include <string_view>

int main() {
    struct test_case { std::string input; std::string expected; };
    const test_case cases[] = {
        {"", "\"\""},
        {"plain / text", "\"plain / text\""},
        {"\"André\"", "\"\\\"André\\\"\""},
        {"path\\file\tline\nnext\r\b\f", "\"path\\\\file\\tline\\nnext\\r\\b\\f\""},
        {std::string("a\0b", 3), "\"a\\u0000b\""},
        {std::string("\x01\x1f", 2), "\"\\u0001\\u001f\""},
    };
    for (const auto & c : cases) {
        if (pf::cli::json_quote(c.input) != c.expected) {
            std::fprintf(stderr, "JSON escaping mismatch\n");
            return 1;
        }
    }
    // Use an explicit span length: do not consume the remainder of the input.
    if (pf::cli::json_quote(std::string_view("quoted\"tail", 6)) != "\"quoted\"") return 1;
    // Every ASCII control byte must have an escaped representation.
    for (int i = 0; i < 32; ++i) {
        const char c = static_cast<char>(i);
        const std::string quoted = pf::cli::json_quote(std::string_view(&c, 1));
        if (quoted.size() < 4 || quoted[1] != '\\') return 1;
    }
    return 0;
}
