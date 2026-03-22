#include "binance_rest.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

/* stub implementation -- real version needs libcurl + openssl */

int binance_init(BinanceClient *client, const char *api_key, const char *secret_key) {
    memset(client, 0, sizeof(*client));
    if (api_key) strncpy(client->api_key, api_key, sizeof(client->api_key) - 1);
    if (secret_key) strncpy(client->secret_key, secret_key, sizeof(client->secret_key) - 1);
    strncpy(client->base_url, "https://api.binance.com", sizeof(client->base_url) - 1);
    client->recv_window = 5000;
    client->verbose = 0;
    return 0;
}

void binance_cleanup(BinanceClient *client) {
    memset(client, 0, sizeof(*client));
}

void binance_set_testnet(BinanceClient *client) {
    strncpy(client->base_url, "https://testnet.binance.vision", sizeof(client->base_url) - 1);
}

int binance_get_klines(
    BinanceClient *client,
    const char *symbol,
    const char *interval,
    int limit,
    Kline *out_klines,
    int *out_count
) {
    (void)client; (void)symbol; (void)interval; (void)limit;
    (void)out_klines;
    *out_count = 0;
    fprintf(stderr, "binance_get_klines: stub -- needs libcurl\n");
    return -1;
}

double binance_get_price(BinanceClient *client, const char *symbol) {
    (void)client; (void)symbol;
    fprintf(stderr, "binance_get_price: stub -- needs libcurl\n");
    return 0.0;
}

int binance_place_order(
    BinanceClient *client,
    const char *symbol,
    OrderSide side,
    OrderType type,
    double price,
    double qty,
    OrderResponse *out_resp
) {
    (void)client; (void)symbol; (void)side; (void)type;
    (void)price; (void)qty;
    memset(out_resp, 0, sizeof(*out_resp));
    fprintf(stderr, "binance_place_order: stub -- needs libcurl + openssl\n");
    return -1;
}

int binance_cancel_order(
    BinanceClient *client,
    const char *symbol,
    int64_t order_id
) {
    (void)client; (void)symbol; (void)order_id;
    fprintf(stderr, "binance_cancel_order: stub -- needs libcurl + openssl\n");
    return -1;
}

int binance_get_margin_account(
    BinanceClient *client,
    MarginAsset *assets,
    int max_assets,
    int *out_count,
    double *total_net_btc
) {
    (void)client; (void)assets; (void)max_assets;
    *out_count = 0;
    *total_net_btc = 0.0;
    fprintf(stderr, "binance_get_margin_account: stub -- needs libcurl + openssl\n");
    return -1;
}

void binance_hmac_sha256(
    const char *key, size_t key_len,
    const char *data, size_t data_len,
    char *out_hex
) {
    (void)key; (void)key_len; (void)data; (void)data_len;
    memset(out_hex, '0', 64);
    out_hex[64] = '\0';
    fprintf(stderr, "binance_hmac_sha256: stub -- needs openssl\n");
}
