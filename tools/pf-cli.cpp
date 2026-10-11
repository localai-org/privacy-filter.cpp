// pf-cli — inspect and run the privacy-filter model.
//   pf-cli --info <model.gguf>
//   pf-cli --model <model.gguf> --tokens <tokens.i32> [--device cpu|cuda|vulkan]
//          [--threads N] [--dump-taps DIR] [--logits-out FILE]
#include "gguf_loader.h"
#include "pf.h"
#include "model.h"
#include "ner.h"
#include "tokenizer.h"
#include "json.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static int info(const char * path) {
    pf::model_file mf;
    if (!mf.open(path, /*with_data =*/ false)) {
        std::fprintf(stderr, "error: %s\n", mf.error.c_str());
        return 1;
    }
    const pf::hparams & h = mf.hp;
    std::printf("arch            openai-privacy-filter\n");
    std::printf("kv pairs        %lld\n", (long long) gguf_get_n_kv(mf.guf));
    std::printf("tensors         %lld\n", (long long) gguf_get_n_tensors(mf.guf));
    std::printf("layers          %d\n", h.n_layer);
    std::printf("d_model         %d\n", h.n_embd);
    std::printf("heads           %d q / %d kv, head_dim %d\n", h.n_head, h.n_head_kv, h.n_rot);
    std::printf("experts         %d, top-%d, ff %d\n", h.n_expert, h.n_expert_used, h.n_ff_exp);
    std::printf("swa radius      %d (band diameter %d)\n", h.swa_radius, 2 * h.swa_radius);
    std::printf("rope            theta %.0f, yarn x%.0f (beta %.0f/%.0f, orig_ctx %d, truncate %s)\n",
                h.rope_freq_base, h.yarn_factor, h.yarn_beta_fast, h.yarn_beta_slow,
                h.n_ctx_orig, h.yarn_truncate ? "true" : "false");
    std::printf("rms_eps         %g\n", h.rms_eps);
    std::printf("labels          %d (%s ... %s)\n", h.n_cls,
                mf.labels.front().c_str(), mf.labels.back().c_str());
    return 0;
}

static bool read_i32(const std::string & path, std::vector<int32_t> & out) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    out.resize(sz / sizeof(int32_t));
    size_t got = std::fread(out.data(), sizeof(int32_t), out.size(), f);
    std::fclose(f);
    return got == out.size();
}

static bool dump_taps(const pf::tap_map & taps, int64_t n_tok, const std::string & dir) {
    std::string meta = "{\n \"n_tok\": " + std::to_string(n_tok) + ",\n \"taps\": {";
    bool first = true;
    for (const auto & [name, d] : taps) {
        const bool is_i32 = !d.i32.empty();
        const std::string path = dir + "/" + name + (is_i32 ? ".i32" : ".f32");
        FILE * f = std::fopen(path.c_str(), "wb");
        if (!f) {
            std::fprintf(stderr, "cannot write %s\n", path.c_str());
            return false;
        }
        if (is_i32) std::fwrite(d.i32.data(), sizeof(int32_t), d.i32.size(), f);
        else        std::fwrite(d.f32.data(), sizeof(float),   d.f32.size(), f);
        std::fclose(f);
        meta += std::string(first ? "" : ",") + "\n  \"" + name + "\": {\"shape\": [" +
                std::to_string(d.n_rows) + ", " + std::to_string(d.n_cols) +
                "], \"dtype\": \"" + (is_i32 ? "i32" : "f32") + "\"}";
        first = false;
    }
    meta += "\n }\n}\n";
    FILE * f = std::fopen((dir + "/meta.json").c_str(), "wb");
    if (!f) return false;
    std::fwrite(meta.data(), 1, meta.size(), f);
    std::fclose(f);
    return true;
}

// Differential-test workhorse: read a pack of texts ([u32 n] then per text
// [u32 len][bytes]), tokenize each (vocab only, no weights), write a pack of
// results ([u32 n_tok][ids i32 x n][offsets i32 x 2n] per text).
static int tok_batch(const char * gguf, const char * in_path, const char * out_path) {
    pf::model_file mf;
    if (!mf.open(gguf, /*with_data =*/ false)) {
        std::fprintf(stderr, "%s\n", mf.error.c_str());
        return 1;
    }
    pf::tokenizer tk;
    if (!tk.load(mf.guf)) {
        std::fprintf(stderr, "%s\n", tk.error.c_str());
        return 1;
    }
    FILE * in = std::fopen(in_path, "rb");
    FILE * out = std::fopen(out_path, "wb");
    if (!in || !out) {
        std::fprintf(stderr, "cannot open %s / %s\n", in_path, out_path);
        return 1;
    }
    uint32_t n_texts = 0;
    if (std::fread(&n_texts, 4, 1, in) != 1) return 1;
    std::vector<char> buf;
    for (uint32_t i = 0; i < n_texts; i++) {
        uint32_t len = 0;
        if (std::fread(&len, 4, 1, in) != 1) return 1;
        buf.resize(len);
        if (len && std::fread(buf.data(), 1, len, in) != len) return 1;
        const auto toks = tk.encode(buf.data(), len);
        const uint32_t n = (uint32_t) toks.size();
        std::fwrite(&n, 4, 1, out);
        for (const auto & t : toks) std::fwrite(&t.id, 4, 1, out);
        for (const auto & t : toks) {
            std::fwrite(&t.start, 4, 1, out);
            std::fwrite(&t.end, 4, 1, out);
        }
    }
    std::fclose(in);
    std::fclose(out);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc == 3 && std::strcmp(argv[1], "--info") == 0) {
        return info(argv[2]);
    }
    if (argc == 5 && std::strcmp(argv[1], "--tok-batch") == 0) {
        return tok_batch(argv[2], argv[3], argv[4]);
    }
    // full pipeline through the public C API: pf-cli --classify <model> <<< text
    if ((argc == 4 || argc == 5) && std::strcmp(argv[1], "--classify") == 0) {
        std::string text;
        char buf[65536];
        size_t r;
        while ((r = std::fread(buf, 1, sizeof(buf), stdin)) > 0) text.append(buf, r);
        if (!text.empty() && text.back() == '\n') text.pop_back();

        pf_ctx * ctx = pf_load(argv[2], argc == 5 ? argv[4] : nullptr, 0);
        if (pf_last_error(ctx)) {
            std::fprintf(stderr, "load: %s\n", pf_last_error(ctx));
            return 1;
        }
        pf_entity * ents = nullptr;
        size_t n_ents = 0;
        const float thr = argc >= 4 ? (float) std::atof(argv[3]) : 0.0f;
        if (pf_classify(ctx, text.data(), text.size(), thr, &ents, &n_ents) != 0) {
            std::fprintf(stderr, "classify: %s\n", pf_last_error(ctx));
            return 1;
        }
        std::printf("[");
        for (size_t i = 0; i < n_ents; i++) {
            std::printf("%s\n {\"entity_group\": %s, \"start\": %d, \"end\": %d, "
                        "\"score\": %.4f, \"text\": %s}",
                        i ? "," : "", pf::cli::json_quote(ents[i].label).c_str(),
                        ents[i].start, ents[i].end, ents[i].score,
                        pf::cli::json_quote(std::string_view(text.data() + ents[i].start,
                            ents[i].end - ents[i].start)).c_str());
        }
        std::printf("\n]\n");
        pf_entities_free(ents, n_ents);
        pf_free(ctx);
        return 0;
    }

    std::string model_path, tokens_path, taps_dir, logits_path, offsets_path, text_path;
    std::string device = "cpu";
    int threads = 0, window = 4096;
    float threshold = 0.0f;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string a = argv[i];
        if      (a == "--model")      model_path   = argv[i + 1];
        else if (a == "--tokens")     tokens_path  = argv[i + 1];
        else if (a == "--dump-taps")  taps_dir     = argv[i + 1];
        else if (a == "--logits-out") logits_path  = argv[i + 1];
        else if (a == "--offsets")    offsets_path = argv[i + 1];
        else if (a == "--text")       text_path    = argv[i + 1];
        else if (a == "--device")     device       = argv[i + 1];
        else if (a == "--threads")    threads      = std::atoi(argv[i + 1]);
        else if (a == "--window")     window       = std::atoi(argv[i + 1]);
        else if (a == "--threshold")  threshold    = (float) std::atof(argv[i + 1]);
        else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
    }
    if (model_path.empty() || tokens_path.empty()) {
        std::fprintf(stderr,
            "usage: pf-cli --info <model.gguf>\n"
            "       pf-cli --model <model.gguf> --tokens <tokens.i32> [--device cpu|cuda|vulkan]\n"
            "              [--threads N] [--window N] [--dump-taps DIR] [--logits-out FILE]\n"
            "              [--offsets <offsets.i32> --text <text.txt> [--threshold F]]\n");
        return 2;
    }

    std::vector<int32_t> ids;
    if (!read_i32(tokens_path, ids)) {
        std::fprintf(stderr, "cannot read %s\n", tokens_path.c_str());
        return 1;
    }

    pf::model m;
    if (!m.load(model_path, device, threads)) {
        std::fprintf(stderr, "load error: %s\n", m.error.c_str());
        return 1;
    }
    std::fprintf(stderr, "loaded %s on %s, %zu tokens\n",
                 model_path.c_str(), m.be.device.c_str(), ids.size());

    pf::tap_map taps;
    std::vector<float> logits;
    if (!m.forward(ids.data(), (int64_t) ids.size(), logits, taps_dir.empty() ? nullptr : &taps)) {
        std::fprintf(stderr, "forward error: %s\n", m.error.c_str());
        return 1;
    }

    if (!taps_dir.empty() && !dump_taps(taps, (int64_t) ids.size(), taps_dir)) {
        return 1;
    }
    if (!logits_path.empty()) {
        FILE * f = std::fopen(logits_path.c_str(), "wb");
        if (!f) { std::fprintf(stderr, "cannot write %s\n", logits_path.c_str()); return 1; }
        std::fwrite(logits.data(), sizeof(float), logits.size(), f);
        std::fclose(f);
    }

    // Span mode: byte offsets from file (the tokenizer lands in P4), text for
    // span extraction. Mirrors the LocalAI TokenClassify response assembly.
    if (!offsets_path.empty() && !text_path.empty()) {
        std::vector<int32_t> offs;
        std::vector<char> text;
        FILE * tf = std::fopen(text_path.c_str(), "rb");
        if (!read_i32(offsets_path, offs) || !tf) {
            std::fprintf(stderr, "cannot read offsets/text\n");
            return 1;
        }
        std::fseek(tf, 0, SEEK_END);
        text.resize(std::ftell(tf));
        std::fseek(tf, 0, SEEK_SET);
        if (std::fread(text.data(), 1, text.size(), tf) != text.size()) { std::fclose(tf); return 1; }
        std::fclose(tf);

        std::vector<pf::ner::tok_span> spans;
        std::string err;
        if (!pf::ner::classify_tokens(m, ids.data(), (int) ids.size(), window, threshold, spans, err)) {
            std::fprintf(stderr, "classify error: %s\n", err.c_str());
            return 1;
        }
        const pf::ner::label_table lt = pf::ner::build_label_table(m.file.labels);
        std::printf("[");
        bool first = true;
        for (const auto & sp : spans) {
            int bstart = offs[2 * sp.tok_begin];
            int bend   = offs[2 * sp.tok_end + 1];
            if (bstart < 0 || bend > (int) text.size() || bstart >= bend) continue;
            // trim leading/trailing ASCII whitespace (o200k folds the leading
            // space into the piece; mask the trimmed form)
            while (bstart < bend && (unsigned char) text[bstart] <= ' ') bstart++;
            while (bend > bstart && (unsigned char) text[bend - 1] <= ' ') bend--;
            if (bstart >= bend) continue;
            std::printf("%s\n {\"entity_group\": %s, \"start\": %d, \"end\": %d, "
                        "\"score\": %.4f, \"text\": %s}",
                        first ? "" : ",", pf::cli::json_quote(lt.category_name(sp.cat)).c_str(),
                        bstart, bend, sp.score,
                        pf::cli::json_quote(std::string_view(text.data() + bstart,
                            bend - bstart)).c_str());
            first = false;
        }
        std::printf("\n]\n");
        return 0;
    }

    // print argmax label per token as a quick eyeball
    const int n_cls = m.hp().n_cls;
    for (size_t t = 0; t < ids.size() && t < 32; t++) {
        int best = 0;
        for (int c = 1; c < n_cls; c++) {
            if (logits[t * n_cls + c] > logits[t * n_cls + best]) best = c;
        }
        std::printf("%3zu %-6d %s\n", t, ids[t], m.file.labels[best].c_str());
    }
    return 0;
}
