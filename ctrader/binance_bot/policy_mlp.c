#include "policy_mlp.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* LayerNorm epsilon matching torch.nn.LayerNorm default. */
#define CTRDPOL_LN_EPS 1e-5f

static int read_header(FILE *f, CtrdpolHeader *hdr) {
    unsigned char raw[CTRDPOL_HEADER_SIZE];
    if (fread(raw, 1, CTRDPOL_HEADER_SIZE, f) != CTRDPOL_HEADER_SIZE) {
        fprintf(stderr, "ctrdpol: short read on header\n");
        return -1;
    }
    if (memcmp(raw, CTRDPOL_MAGIC, 8) != 0) {
        fprintf(stderr, "ctrdpol: bad magic (expected %s)\n", CTRDPOL_MAGIC);
        return -1;
    }
    uint32_t version;
    memcpy(&version, raw + 8, 4);
    if (version != CTRDPOL_VERSION) {
        fprintf(stderr, "ctrdpol: unsupported version %u (want %u)\n", version, CTRDPOL_VERSION);
        return -1;
    }
    memcpy(&hdr->obs_dim,          raw + 12, 4);
    memcpy(&hdr->hidden,           raw + 16, 4);
    memcpy(&hdr->n_encoder_layers, raw + 20, 4);
    memcpy(&hdr->use_encoder_norm, raw + 24, 4);
    memcpy(&hdr->actor_hidden,     raw + 28, 4);
    memcpy(&hdr->num_actions,      raw + 32, 4);
    memcpy(&hdr->num_symbols,      raw + 36, 4);
    memcpy(&hdr->activation,       raw + 40, 4);
    memcpy(&hdr->disable_shorts,   raw + 44, 4);
    return 0;
}

static float *read_floats(FILE *f, size_t n) {
    float *buf = (float *)malloc(n * sizeof(float));
    if (!buf) {
        fprintf(stderr, "ctrdpol: oom (%zu floats)\n", n);
        return NULL;
    }
    size_t got = fread(buf, sizeof(float), n, f);
    if (got != n) {
        fprintf(stderr, "ctrdpol: short read (%zu/%zu floats)\n", got, n);
        free(buf);
        return NULL;
    }
    return buf;
}

int ctrdpol_load(CtrdpolPolicy *pol, const char *path) {
    memset(pol, 0, sizeof(*pol));
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "ctrdpol: cannot open %s: %s\n", path, strerror(errno));
        return -1;
    }

    if (read_header(f, &pol->hdr) != 0) { fclose(f); return -1; }

    if (pol->hdr.activation != CTRDPOL_ACT_RELU) {
        fprintf(stderr, "ctrdpol: activation %u not supported (only relu=0)\n",
                pol->hdr.activation);
        fclose(f); return -1;
    }
    if (pol->hdr.n_encoder_layers == 0 || pol->hdr.n_encoder_layers > 8) {
        fprintf(stderr, "ctrdpol: bogus n_encoder_layers=%u\n", pol->hdr.n_encoder_layers);
        fclose(f); return -1;
    }

    const uint32_t H = pol->hdr.hidden;
    const uint32_t A = pol->hdr.actor_hidden;
    const uint32_t N = pol->hdr.num_actions;

    pol->encoder_weights = (float **)calloc(pol->hdr.n_encoder_layers, sizeof(float *));
    pol->encoder_biases  = (float **)calloc(pol->hdr.n_encoder_layers, sizeof(float *));
    if (!pol->encoder_weights || !pol->encoder_biases) goto fail;

    for (uint32_t li = 0; li < pol->hdr.n_encoder_layers; li++) {
        uint32_t in_dim = (li == 0) ? pol->hdr.obs_dim : H;
        pol->encoder_weights[li] = read_floats(f, (size_t)H * in_dim);
        if (!pol->encoder_weights[li]) goto fail;
        pol->encoder_biases[li] = read_floats(f, H);
        if (!pol->encoder_biases[li]) goto fail;
    }

    if (pol->hdr.use_encoder_norm) {
        pol->encoder_norm_weight = read_floats(f, H);
        pol->encoder_norm_bias   = read_floats(f, H);
        if (!pol->encoder_norm_weight || !pol->encoder_norm_bias) goto fail;
    }

    pol->actor0_weight = read_floats(f, (size_t)A * H);
    pol->actor0_bias   = read_floats(f, A);
    pol->actor2_weight = read_floats(f, (size_t)N * A);
    pol->actor2_bias   = read_floats(f, N);
    if (!pol->actor0_weight || !pol->actor0_bias || !pol->actor2_weight || !pol->actor2_bias) {
        goto fail;
    }

    /* Sanity: file should be exhausted now. */
    unsigned char tail;
    if (fread(&tail, 1, 1, f) != 0) {
        fprintf(stderr, "ctrdpol: warning, %s has trailing bytes after expected payload\n", path);
    }
    fclose(f);

    /* Scratch allocations. */
    pol->scratch_enc_in  = (float *)calloc(pol->hdr.obs_dim, sizeof(float));
    pol->scratch_enc_a   = (float *)calloc(H, sizeof(float));
    pol->scratch_enc_b   = (float *)calloc(H, sizeof(float));
    pol->scratch_actor   = (float *)calloc(A, sizeof(float));
    pol->scratch_logits  = (float *)calloc(N, sizeof(float));
    if (!pol->scratch_enc_in || !pol->scratch_enc_a || !pol->scratch_enc_b ||
        !pol->scratch_actor || !pol->scratch_logits) {
        fprintf(stderr, "ctrdpol: oom allocating scratch\n");
        ctrdpol_free(pol);
        return -1;
    }

    return 0;

fail:
    fclose(f);
    ctrdpol_free(pol);
    return -1;
}

void ctrdpol_free(CtrdpolPolicy *pol) {
    if (pol->encoder_weights) {
        for (uint32_t li = 0; li < pol->hdr.n_encoder_layers; li++) {
            free(pol->encoder_weights[li]);
        }
        free(pol->encoder_weights);
    }
    if (pol->encoder_biases) {
        for (uint32_t li = 0; li < pol->hdr.n_encoder_layers; li++) {
            free(pol->encoder_biases[li]);
        }
        free(pol->encoder_biases);
    }
    free(pol->encoder_norm_weight);
    free(pol->encoder_norm_bias);
    free(pol->actor0_weight);
    free(pol->actor0_bias);
    free(pol->actor2_weight);
    free(pol->actor2_bias);
    free(pol->scratch_enc_in);
    free(pol->scratch_enc_a);
    free(pol->scratch_enc_b);
    free(pol->scratch_actor);
    free(pol->scratch_logits);
    memset(pol, 0, sizeof(*pol));
}

/* y = W @ x + b
 *   W is [out × in], row-major (matches PyTorch Linear.weight layout)
 *   x is [in], y is [out]. b may be NULL (then no bias is added).
 */
static void linear(const float *W, const float *b, const float *x,
                   float *y, int out_dim, int in_dim) {
    for (int o = 0; o < out_dim; o++) {
        float acc = (b ? b[o] : 0.0f);
        const float *row = W + (size_t)o * in_dim;
        /* simple dot product; compiler will auto-vectorise with -O2/-O3. */
        for (int i = 0; i < in_dim; i++) acc += row[i] * x[i];
        y[o] = acc;
    }
}

static void relu_inplace(float *v, int n) {
    for (int i = 0; i < n; i++) {
        if (v[i] < 0.0f) v[i] = 0.0f;
    }
}

/* torch.nn.LayerNorm(hidden) with elementwise affine (weight + bias). */
static void layernorm_inplace(float *v, const float *w, const float *b, int n) {
    double mean = 0.0;
    for (int i = 0; i < n; i++) mean += v[i];
    mean /= (double)n;
    double var = 0.0;
    for (int i = 0; i < n; i++) {
        double d = (double)v[i] - mean;
        var += d * d;
    }
    var /= (double)n;
    double inv_std = 1.0 / sqrt(var + (double)CTRDPOL_LN_EPS);
    for (int i = 0; i < n; i++) {
        double normed = ((double)v[i] - mean) * inv_std;
        v[i] = (float)(normed * (double)w[i] + (double)b[i]);
    }
}

int ctrdpol_forward(const CtrdpolPolicy *pol, const float *obs_f32, float *out_logits) {
    const uint32_t H = pol->hdr.hidden;
    const uint32_t A = pol->hdr.actor_hidden;
    const uint32_t N = pol->hdr.num_actions;

    /* Ping-pong buffers between enc_a and enc_b so we can do any number of
     * encoder layers without extra allocation. */
    float *cur_in  = (float *)obs_f32;    /* first layer reads from caller's obs */
    int   cur_in_dim = (int)pol->hdr.obs_dim;
    float *cur_out = pol->scratch_enc_a;
    float *other   = pol->scratch_enc_b;

    for (uint32_t li = 0; li < pol->hdr.n_encoder_layers; li++) {
        linear(pol->encoder_weights[li], pol->encoder_biases[li],
               cur_in, cur_out, (int)H, cur_in_dim);
        relu_inplace(cur_out, (int)H);
        cur_in = cur_out;
        cur_in_dim = (int)H;
        /* swap so next iter writes to the other buffer */
        float *tmp = cur_out;
        cur_out = other;
        other = tmp;
    }

    /* cur_in now holds encoder output, shape [H]. */
    if (pol->hdr.use_encoder_norm) {
        layernorm_inplace(cur_in, pol->encoder_norm_weight, pol->encoder_norm_bias, (int)H);
    }

    linear(pol->actor0_weight, pol->actor0_bias, cur_in, pol->scratch_actor, (int)A, (int)H);
    relu_inplace(pol->scratch_actor, (int)A);
    linear(pol->actor2_weight, pol->actor2_bias, pol->scratch_actor, out_logits, (int)N, (int)A);
    return 0;
}

int ctrdpol_argmax(const float *logits, int n) {
    if (n <= 0) return -1;
    int best = 0;
    float bv = logits[0];
    for (int i = 1; i < n; i++) {
        if (logits[i] > bv) { bv = logits[i]; best = i; }
    }
    return best;
}
