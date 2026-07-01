#include "backend.h"

#include <ggml-backend.h>
#include <ggml-cpu.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <thread>
#include <vector>

namespace pf {

namespace {

std::string lower(std::string s) {
    for (char & c : s) c = (char) std::tolower((unsigned char) c);
    return s;
}

// Default CPU threads = physical cores. SMT siblings only add contention for
// matmul-heavy work, so on x86 (logical == 2x physical) we halve -- but ONLY
// when SMT is actually on. ARM / Apple silicon have no SMT, where a blanket /2
// silently throws away half the cores.
unsigned default_cpu_threads() {
    unsigned logical = std::max(1u, std::thread::hardware_concurrency());
    std::ifstream smt("/sys/devices/system/cpu/smt/active");
    int on = 0;
    if (smt >> on && on == 1) {
        return std::max(1u, logical / 2);
    }
    return logical;
}

// Both discrete and integrated GPUs are valid offload targets. ggml reports
// iGPUs (e.g. AMD Radeon 780M / any uma:1 device) as a distinct device type,
// so a plain "== GPU" check silently drops them and llama.cpp-capable hardware
// looks unsupported.
bool is_gpu_device(ggml_backend_dev_t dev) {
    const enum ggml_backend_dev_type t = ggml_backend_dev_type(dev);
    return t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU;
}

// "cuda:1" -> ("cuda", 1, true); "vulkan" -> ("vulkan", 0, false); "" -> ("", 0, false).
// `explicit_index` distinguishes a bare "vulkan" (auto-pick best) from "vulkan:0"
// (literal device index) -- they'd otherwise both yield index 0.
void parse_device(const std::string & req, std::string & name, int & index, bool & explicit_index) {
    const size_t colon = req.find(':');
    name  = lower(colon == std::string::npos ? req : req.substr(0, colon));
    explicit_index = (colon != std::string::npos);
    index = explicit_index ? std::atoi(req.c_str() + colon + 1) : 0;
}

} // namespace

// Discover dynamically-loadable backends once. For a GGML_BACKEND_DL +
// GGML_CPU_ALL_VARIANTS build this loads every libggml-cpu-<isa>.so and ggml
// scores them, so the host's best ISA (e.g. zen4/AVX-512) is selected at run
// time -- the portable way to ship SIMD without baking -march into one binary
// (and without Nix's wrapper silently dropping -march=native). No-op / harmless
// for a statically-linked build, where backends register at static-init.
static void load_backends_once() {
    static const bool done = [] { ggml_backend_load_all(); return true; }();
    (void) done;
}

// Set threads through the backend registry rather than ggml_backend_cpu_set_n_threads:
// in a DL build that symbol lives in the variant .so, not in the linked base.
static void set_cpu_threads(ggml_backend_t be, int n_threads) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(be));
    auto set_fn = (ggml_backend_set_n_threads_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    if (set_fn) set_fn(be, n_threads);
}

bool engine_backend::init(const std::string & device_req, int n_threads) {
    release();
    load_backends_once();

    std::string name;
    int         want_idx = 0;
    bool        explicit_idx = false;
    parse_device(device_req, name, want_idx, explicit_idx);

    if (name.empty() || name == "cpu") {
        be = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (!be) {
            error = "CPU backend init failed";
            return false;
        }
        device = "cpu";
        if (const char * env = std::getenv("PF_NTHREADS")) {
            // explicit override (tuning / benchmarking); 0 falls through to auto
            if (int v = std::atoi(env)) n_threads = v;
        }
        if (n_threads <= 0) {
            n_threads = (int) default_cpu_threads();
        }
        set_cpu_threads(be, n_threads);
    } else if (name == "gpu" || name == "cuda" || name == "vulkan") {
        // "gpu" picks the best GPU of whichever backend was compiled in;
        // "cuda"/"vulkan" pin a specific backend when more than one is built.
        const std::string want_reg = (name == "gpu") ? "" : name;

        // Gather matching GPU/iGPU devices in ggml enumeration order.
        std::vector<ggml_backend_dev_t> matches;
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (!is_gpu_device(dev)) continue;
            if (!want_reg.empty()) {
                const char * reg = ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev));
                if (!reg || lower(reg) != want_reg) continue;
            }
            matches.push_back(dev);
        }

        // Try candidates in priority order until one initializes.
        //   "vulkan:N" -> the literal Nth match (user pinned it).
        //   bare "vulkan"/"gpu" -> prefer a *discrete* GPU, fall back to an
        //   integrated one. ggml keeps devices in driver-enumeration order,
        //   which frequently lists the iGPU first (e.g. RADV on an AMD APU +
        //   dGPU box), so taking index 0 would hand a fast dGPU host its slow
        //   iGPU. iGPU-only machines still resolve via the fallback.
        std::vector<ggml_backend_dev_t> candidates;
        if (explicit_idx) {
            if (want_idx >= 0 && want_idx < (int) matches.size()) {
                candidates.push_back(matches[want_idx]);
            }
        } else {
            for (ggml_backend_dev_t d : matches) {
                if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_GPU) candidates.push_back(d);
            }
            for (ggml_backend_dev_t d : matches) {
                if (ggml_backend_dev_type(d) == GGML_BACKEND_DEVICE_TYPE_IGPU) candidates.push_back(d);
            }
        }

        for (ggml_backend_dev_t d : candidates) {
            be = ggml_backend_dev_init(d, nullptr);
            if (be) {
                device = ggml_backend_dev_name(d);
                break;
            }
        }
        if (!be) {
            error = "no usable '" + name + "' device (built with PF_CUDA/PF_VULKAN? driver present?)";
            return false;
        }
    } else {
        error = "unknown device '" + device_req + "' (want cpu|gpu|cuda|vulkan)";
        return false;
    }

    galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(be));
    if (!galloc) {
        error = "gallocr init failed";
        release();
        return false;
    }
    return true;
}

void engine_backend::release() {
    if (galloc) { ggml_gallocr_free(galloc); galloc = nullptr; }
    if (be)     { ggml_backend_free(be);     be = nullptr; }
    device.clear();
}

bool engine_backend::is_cpu() const {
    return be && ggml_backend_dev_type(ggml_backend_get_device(be)) == GGML_BACKEND_DEVICE_TYPE_CPU;
}

} // namespace pf
