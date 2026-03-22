#ifndef MKTD_READER_H
#define MKTD_READER_H

#define MKTD_MAX_SYMBOLS 64
#define MKTD_SYM_NAME_LEN 16
#define MKTD_HEADER_SIZE 64
#define MKTD_PRICE_FEATS 5

typedef struct {
    int num_symbols;
    int num_timesteps;
    int features_per_sym;
    char symbol_names[MKTD_MAX_SYMBOLS][MKTD_SYM_NAME_LEN];
    float *features;   /* [T * S * features_per_sym] */
    float *opens;      /* [T * S] */
    float *highs;      /* [T * S] */
    float *lows;       /* [T * S] */
    float *closes;     /* [T * S] */
    float *volumes;    /* [T * S] */
    unsigned char *tradable; /* [T * S] or NULL */
    void *_file_buf;
} MktdData;

int mktd_load(const char *path, MktdData *data);
void mktd_free(MktdData *data);

#endif
