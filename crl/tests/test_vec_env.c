#include "../vec_env.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

/* Write a minimal synthetic MKTD binary for testing */
static void write_test_data(const char* path) {
    int S = 2, T = 100, F = 16;
    FILE* fp = fopen(path, "wb");
    assert(fp);

    /* header */
    DataHeader hdr;
    memset(&hdr, 0, sizeof(hdr));
    memcpy(hdr.magic, "MKTD", 4);
    hdr.version = 2;
    hdr.num_symbols = S;
    hdr.num_timesteps = T;
    hdr.features_per_sym = F;
    hdr.price_features = 5;
    fwrite(&hdr, sizeof(hdr), 1, fp);

    /* symbol names */
    char names[2][16] = {"SYM0", "SYM1"};
    fwrite(names, 16, 2, fp);

    /* features [T][S][F] */
    for (int t = 0; t < T; t++) {
        for (int s = 0; s < S; s++) {
            for (int f = 0; f < F; f++) {
                float v = (float)(t * S * F + s * F + f) * 0.001f;
                fwrite(&v, sizeof(float), 1, fp);
            }
        }
    }

    /* prices [T][S][5] -- synthetic: close=100+t*0.1, open=close-0.05, high=close+0.5, low=close-0.5, vol=1000 */
    for (int t = 0; t < T; t++) {
        for (int s = 0; s < S; s++) {
            float close = 100.0f + t * 0.1f + s * 10.0f;
            float open = close - 0.05f;
            float high = close + 0.5f;
            float low = close - 0.5f;
            float vol = 1000.0f;
            fwrite(&open, sizeof(float), 1, fp);
            fwrite(&high, sizeof(float), 1, fp);
            fwrite(&low, sizeof(float), 1, fp);
            fwrite(&close, sizeof(float), 1, fp);
            fwrite(&vol, sizeof(float), 1, fp);
        }
    }

    /* tradable mask [T][S] */
    for (int t = 0; t < T; t++) {
        for (int s = 0; s < S; s++) {
            unsigned char v = 1;
            fwrite(&v, 1, 1, fp);
        }
    }

    fclose(fp);
}

int main() {
    const char* test_data = "/tmp/crl_test_data.bin";
    write_test_data(test_data);

    VecEnvConfig cfg = {0};
    cfg.max_steps = 50;
    cfg.fee_rate = 0.001f;
    cfg.max_leverage = 1.0f;
    cfg.periods_per_year = 8760.0f;
    cfg.action_allocation_bins = 2;
    cfg.action_level_bins = 2;
    cfg.action_max_offset_bps = 10.0f;
    cfg.reward_scale = 10.0f;
    cfg.reward_clip = 5.0f;
    cfg.cash_penalty = 0.01f;
    cfg.fill_probability = 1.0f;
    cfg.smooth_downside_temperature = 0.02f;

    int num_envs = 4;
    VecEnv* ve = vec_env_create(test_data, num_envs, &cfg);
    assert(ve != NULL);
    assert(ve->num_envs == num_envs);
    assert(ve->obs_size == 2 * 16 + 5 + 2); /* S=2, F=16 */

    printf("obs_size=%d num_actions=%d\n", ve->obs_size, ve->num_actions);

    /* test reset */
    vec_env_reset(ve, 42);
    for (int i = 0; i < num_envs; i++) {
        assert(ve->done_buf[i] == 0);
        /* check obs is non-zero (features should be populated) */
        float obs_sum = 0;
        for (int j = 0; j < ve->obs_size; j++) {
            obs_sum += fabsf(ve->obs_buf[i * ve->obs_size + j]);
        }
        assert(obs_sum > 0.0f);
    }

    /* test step with action=0 (flat) for 10 steps */
    for (int step = 0; step < 10; step++) {
        for (int i = 0; i < num_envs; i++) {
            ve->act_buf[i] = 0;
        }
        vec_env_step(ve);
    }

    /* test step with action=1 (first long action) */
    for (int i = 0; i < num_envs; i++) {
        ve->act_buf[i] = 1;
    }
    vec_env_step(ve);

    /* run until some env terminates */
    int total_steps = 0;
    int any_done = 0;
    while (total_steps < 200 && !any_done) {
        for (int i = 0; i < num_envs; i++) {
            ve->act_buf[i] = rand() % ve->num_actions;
        }
        vec_env_step(ve);
        total_steps++;
        for (int i = 0; i < num_envs; i++) {
            if (ve->done_buf[i]) any_done = 1;
        }
    }

    /* check logs for completed episodes */
    float total_n = 0;
    for (int i = 0; i < num_envs; i++) {
        Log log;
        vec_env_get_log(ve, i, &log);
        total_n += log.n;
    }
    printf("Ran %d steps, completed %.0f episodes\n", total_steps, total_n);
    assert(total_n > 0);

    /* test forced offset */
    vec_env_set_offset(ve, 0, 10);
    c_reset(&ve->envs[0]);
    assert(ve->envs[0].agent.data_offset == 10);

    vec_env_free(ve);
    remove(test_data);

    printf("ALL VEC_ENV TESTS PASSED\n");
    return 0;
}
