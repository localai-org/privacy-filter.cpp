// Prototype: O(n*band) block-local sliding-window attention vs the full O(n^2)
// masked attention, on random data, to validate correctness before wiring it
// into the model. Single head for clarity.
//
//   Full:   scores[n_kv,n_q] = mul_mat(K,Q); mask |q-k|<=r; softmax; out=V^T@scores
//   Banded: tokens grouped into blocks of B (>= r). Each query block attends only
//           to blocks {i-1,i,i+1} (3B keys, since r<=B). A per-block mask
//           [3B,B,n_blocks] (O(n*B) memory) carries the band + edge validity.
#include <ggml.h>
#include <ggml-cpu.h>
#include <ggml-backend.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>

static ggml_backend_t cpu() {
    static ggml_backend_t be = [] { ggml_backend_load_all();
        return ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr); }();
    return be;
}
static void run(ggml_context * ctx, ggml_tensor * out) {
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_graph_compute(cpu(), gf);
}

int main(int argc, char ** argv) {
    const int d = 64;
    const int r = 128;                       // sliding-window radius
    const int B = argc > 1 ? std::atoi(argv[1]) : 256;   // block size (>= r)
    const int n = argc > 2 ? std::atoi(argv[2]) : 4096;  // tokens (multiple of B)
    const int nb = n / B;
    const float scale = 1.0f / std::sqrt((float) d);

    ggml_init_params p = { (size_t) 2048 * 1024 * 1024, nullptr, false };
    ggml_context * ctx = ggml_init(p);

    // shared random q,k,v  [d, n]
    ggml_tensor * Q = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, n);
    ggml_tensor * K = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, n);
    ggml_tensor * V = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d, n);
    for (ggml_tensor * t : { Q, K, V })
        for (int i = 0; i < d * n; i++) ((float *) t->data)[i] = (float) (rand() % 2000) / 1000.0f - 1.0f;

    // ---- full reference ----
    ggml_tensor * Fmask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n, n);  // [n_kv, n_q]
    for (int q = 0; q < n; q++)
        for (int k = 0; k < n; k++)
            ((float *) Fmask->data)[(size_t) q * n + k] = (std::abs(q - k) <= r) ? 0.0f : -INFINITY;
    ggml_tensor * sc = ggml_mul_mat(ctx, K, Q);                          // [n_kv, n_q]
    sc = ggml_soft_max_ext(ctx, sc, Fmask, scale, 0.0f);
    ggml_tensor * Vt = ggml_cont(ctx, ggml_transpose(ctx, V));           // [n, d]
    ggml_tensor * full = ggml_mul_mat(ctx, Vt, sc);                      // [d, n_q]
    run(ctx, full);

    // ---- banded ----
    ggml_tensor * qb = ggml_reshape_3d(ctx, Q, d, B, nb);               // [d, B, nb]
    ggml_tensor * kb = ggml_reshape_3d(ctx, K, d, B, nb);
    ggml_tensor * vb = ggml_reshape_3d(ctx, V, d, B, nb);
    // pad a zero block each side along the block axis, then 3 shifted views
    auto ctx3 = [&](ggml_tensor * x) {
        ggml_tensor * z = ggml_scale(ctx, ggml_view_3d(ctx, x, d, B, 1, x->nb[1], x->nb[2], 0), 0.0f);
        ggml_tensor * pad = ggml_concat(ctx, ggml_concat(ctx, z, x, 2), z, 2);  // [d, B, nb+2]
        ggml_tensor * prev = ggml_view_3d(ctx, pad, d, B, nb, pad->nb[1], pad->nb[2], 0 * pad->nb[2]);
        ggml_tensor * self = ggml_view_3d(ctx, pad, d, B, nb, pad->nb[1], pad->nb[2], 1 * pad->nb[2]);
        ggml_tensor * next = ggml_view_3d(ctx, pad, d, B, nb, pad->nb[1], pad->nb[2], 2 * pad->nb[2]);
        return ggml_cont(ctx, ggml_concat(ctx, ggml_concat(ctx, prev, self, 1), next, 1)); // [d, 3B, nb]
    };
    ggml_tensor * kc = ctx3(kb);                                        // [d, 3B, nb]
    ggml_tensor * vc = ctx3(vb);
    ggml_tensor * Bmask = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 3 * B, B, nb);  // [3B, B, nb]
    for (int bi = 0; bi < nb; bi++)
        for (int j = 0; j < B; j++)
            for (int pp = 0; pp < 3 * B; pp++) {
                const int qpos = bi * B + j;
                const int kpos = bi * B - B + pp;
                const bool vis = std::abs(qpos - kpos) <= r && kpos >= 0 && kpos < n;
                ((float *) Bmask->data)[((size_t) bi * B + j) * (3 * B) + pp] = vis ? 0.0f : -INFINITY;
            }
    ggml_tensor * scb = ggml_mul_mat(ctx, kc, qb);                      // [3B, B, nb]
    scb = ggml_soft_max_ext(ctx, scb, Bmask, scale, 0.0f);
    ggml_tensor * vct = ggml_cont(ctx, ggml_transpose(ctx, vc));        // [3B, d, nb]
    ggml_tensor * outb = ggml_mul_mat(ctx, vct, scb);                  // [d, B, nb]
    outb = ggml_cont_2d(ctx, outb, d, n);                              // [d, n]
    run(ctx, outb);

    // compare
    double maxabs = 0, maxrel = 0;
    for (int i = 0; i < d * n; i++) {
        const double a = ((float *) full->data)[i], b = ((float *) outb->data)[i];
        maxabs = std::max(maxabs, std::fabs(a - b));
        maxrel = std::max(maxrel, std::fabs(a - b) / (std::fabs(a) + 1e-6));
    }
    const double full_mask_mib = (double) n * n * 4 / 1048576.0;
    const double band_mask_mib = (double) 3 * B * B * nb * 4 / 1048576.0;
    std::printf("n=%d B=%d r=%d | max|d|=%.2e maxrel=%.2e | mask: full %.1f MiB, band %.1f MiB (%.1fx)\n",
                n, B, r, maxabs, maxrel, full_mask_mib, band_mask_mib, full_mask_mib / band_mask_mib);
    return 0;
}
