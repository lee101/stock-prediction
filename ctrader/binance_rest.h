#ifndef BINANCE_REST_H
#define BINANCE_REST_H

#include <stdint.h>
#include <stddef.h>

#define BINANCE_MAX_SYMBOL 16
#define BINANCE_MAX_KLINES 1500
#define BINANCE_MAX_RESPONSE (1 << 20)

typedef struct {
    char api_key[128];
    char secret_key[128];
    char base_url[128];
    int recv_window;
    int verbose;
} BinanceClient;

typedef struct {
    int64_t open_time;
    double open;
    double high;
    double low;
    double close;
    double volume;
    int64_t close_time;
} Kline;

typedef struct {
    char symbol[BINANCE_MAX_SYMBOL];
    double free_qty;
    double locked_qty;
    double borrowed_qty;
    double net_asset;
} MarginAsset;

typedef enum {
    ORDER_BUY,
    ORDER_SELL
} OrderSide;

typedef enum {
    ORDER_LIMIT,
    ORDER_MARKET
} OrderType;

typedef struct {
    int64_t order_id;
    char symbol[BINANCE_MAX_SYMBOL];
    OrderSide side;
    OrderType type;
    double price;
    double orig_qty;
    double executed_qty;
    char status[32];
} OrderResponse;

/* init/cleanup */
int binance_init(BinanceClient *client, const char *api_key, const char *secret_key);
void binance_cleanup(BinanceClient *client);
void binance_set_testnet(BinanceClient *client);

/* market data (unauthenticated) */
int binance_get_klines(
    BinanceClient *client,
    const char *symbol,
    const char *interval,
    int limit,
    Kline *out_klines,
    int *out_count
);

double binance_get_price(BinanceClient *client, const char *symbol);

/* trading (authenticated) */
int binance_place_order(
    BinanceClient *client,
    const char *symbol,
    OrderSide side,
    OrderType type,
    double price,
    double qty,
    OrderResponse *out_resp
);

int binance_cancel_order(
    BinanceClient *client,
    const char *symbol,
    int64_t order_id
);

/* margin account */
int binance_get_margin_account(
    BinanceClient *client,
    MarginAsset *assets,
    int max_assets,
    int *out_count,
    double *total_net_btc
);

/* HMAC-SHA256 signature helper */
void binance_hmac_sha256(
    const char *key, size_t key_len,
    const char *data, size_t data_len,
    char *out_hex
);

#endif
