#include "mktd_reader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char magic[4];
    unsigned int version;
    unsigned int num_symbols;
    unsigned int num_timesteps;
    unsigned int features_per_sym;
    unsigned int price_features;
    char padding[40];
} MktdHeader;

int mktd_load(const char *path, MktdData *data) {
    memset(data, 0, sizeof(*data));

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "mktd_load: cannot open %s\n", path);
        return -1;
    }

    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    if (file_size < MKTD_HEADER_SIZE) {
        fprintf(stderr, "mktd_load: file too small (%ld bytes)\n", file_size);
        fclose(fp);
        return -1;
    }

    void *buf = malloc((size_t)file_size);
    if (!buf) {
        fclose(fp);
        return -1;
    }
    if ((long)fread(buf, 1, (size_t)file_size, fp) != file_size) {
        free(buf);
        fclose(fp);
        return -1;
    }
    fclose(fp);

    MktdHeader *hdr = (MktdHeader *)buf;
    if (memcmp(hdr->magic, "MKTD", 4) != 0) {
        fprintf(stderr, "mktd_load: bad magic in %s\n", path);
        free(buf);
        return -1;
    }

    int S = (int)hdr->num_symbols;
    int T = (int)hdr->num_timesteps;
    int F = (hdr->features_per_sym > 0) ? (int)hdr->features_per_sym : 16;

    if (S <= 0 || S > MKTD_MAX_SYMBOLS) {
        fprintf(stderr, "mktd_load: invalid num_symbols %d\n", S);
        free(buf);
        return -1;
    }
    if (T <= 0) {
        fprintf(stderr, "mktd_load: invalid num_timesteps %d\n", T);
        free(buf);
        return -1;
    }

    char *ptr = (char *)buf + MKTD_HEADER_SIZE;
    char *end = (char *)buf + file_size;

    /* symbol names */
    size_t names_bytes = (size_t)S * MKTD_SYM_NAME_LEN;
    if (ptr + names_bytes > end) {
        fprintf(stderr, "mktd_load: truncated symbol names\n");
        free(buf);
        return -1;
    }
    for (int i = 0; i < S; i++) {
        memcpy(data->symbol_names[i], ptr + i * MKTD_SYM_NAME_LEN, MKTD_SYM_NAME_LEN);
        data->symbol_names[i][MKTD_SYM_NAME_LEN - 1] = '\0';
    }
    ptr += names_bytes;

    /* features: float[T][S][F] */
    size_t feat_bytes = (size_t)T * S * F * sizeof(float);
    if (ptr + feat_bytes > end) {
        fprintf(stderr, "mktd_load: truncated features\n");
        free(buf);
        return -1;
    }
    data->features = (float *)ptr;
    ptr += feat_bytes;

    /* prices: float[T][S][5] (OHLCV interleaved) */
    size_t price_bytes = (size_t)T * S * MKTD_PRICE_FEATS * sizeof(float);
    if (ptr + price_bytes > end) {
        fprintf(stderr, "mktd_load: truncated prices\n");
        free(buf);
        return -1;
    }
    float *prices = (float *)ptr;
    ptr += price_bytes;

    /* deinterleave OHLCV into separate arrays */
    size_t ts_count = (size_t)T * S;
    data->opens   = (float *)malloc(ts_count * sizeof(float));
    data->highs   = (float *)malloc(ts_count * sizeof(float));
    data->lows    = (float *)malloc(ts_count * sizeof(float));
    data->closes  = (float *)malloc(ts_count * sizeof(float));
    data->volumes = (float *)malloc(ts_count * sizeof(float));
    if (!data->opens || !data->highs || !data->lows || !data->closes || !data->volumes) {
        free(data->opens); free(data->highs); free(data->lows);
        free(data->closes); free(data->volumes);
        free(buf);
        return -1;
    }

    for (size_t i = 0; i < ts_count; i++) {
        data->opens[i]   = prices[i * MKTD_PRICE_FEATS + 0];
        data->highs[i]   = prices[i * MKTD_PRICE_FEATS + 1];
        data->lows[i]    = prices[i * MKTD_PRICE_FEATS + 2];
        data->closes[i]  = prices[i * MKTD_PRICE_FEATS + 3];
        data->volumes[i] = prices[i * MKTD_PRICE_FEATS + 4];
    }

    /* optional tradable mask: uint8[T][S] */
    size_t mask_bytes = (size_t)T * S;
    if (hdr->version >= 2 && ptr + mask_bytes <= end) {
        data->tradable = (unsigned char *)ptr;
        ptr += mask_bytes;
    } else if (ptr + mask_bytes <= end) {
        data->tradable = (unsigned char *)ptr;
    } else {
        data->tradable = NULL;
    }

    data->num_symbols = S;
    data->num_timesteps = T;
    data->features_per_sym = F;
    data->_file_buf = buf;

    fprintf(stderr, "mktd_load: %d symbols, %d timesteps, %d features from %s\n",
            S, T, F, path);
    for (int i = 0; i < S; i++) {
        fprintf(stderr, "  [%d] %s\n", i, data->symbol_names[i]);
    }

    return 0;
}

void mktd_free(MktdData *data) {
    if (!data) return;
    free(data->opens);
    free(data->highs);
    free(data->lows);
    free(data->closes);
    free(data->volumes);
    free(data->_file_buf);
    memset(data, 0, sizeof(*data));
}
