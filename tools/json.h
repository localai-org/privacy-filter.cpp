// SPDX-License-Identifier: MIT
#pragma once

#include <string>
#include <string_view>

namespace pf::cli {

// Preserve UTF-8 bytes and escape JSON syntax and ASCII control characters.
inline std::string json_quote(std::string_view text) {
    static constexpr char hex[] = "0123456789abcdef";
    std::string out = "\"";
    for (unsigned char c : text) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    out += "\\u00";
                    out += hex[c >> 4];
                    out += hex[c & 0x0f];
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    out += '"';
    return out;
}

} // namespace pf::cli
