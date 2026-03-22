#include "binance_rest.h"
#include "vendor/cJSON.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <curl/curl.h>
#include <openssl/hmac.h>
#include <openssl/evp.h>

typedef struct {
    char *data;
    size_t len;
    size_t cap;
} ResponseBuf;

static size_t write_callback(void *contents, size_t size, size_t nmemb, void *userp) {
    size_t total = size * nmemb;
    ResponseBuf *buf = (ResponseBuf *)userp;
    if (buf->len + total >= buf->cap) return 0;
    memcpy(buf->data + buf->len, contents, total);
    buf->len += total;
    buf->data[buf->len] = '\0';
    return total;
}

static int64_t get_timestamp_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (int64_t)tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

static int http_request(const char *url, const char *method, const char *body,
                        const char *api_key, int verbose,
                        char *response, size_t max_len) {
    CURL *curl = curl_easy_init();
    if (!curl) return -1;

    ResponseBuf buf = { .data = response, .len = 0, .cap = max_len };

    curl_easy_setopt(curl, CURLOPT_URL, url);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buf);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 5L);

    if (verbose) curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);

    struct curl_slist *headers = NULL;
    if (api_key && api_key[0]) {
        char hdr[256];
        snprintf(hdr, sizeof(hdr), "X-MBX-APIKEY: %s", api_key);
        headers = curl_slist_append(headers, hdr);
    }

    if (strcmp(method, "POST") == 0) {
        curl_easy_setopt(curl, CURLOPT_POST, 1L);
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body ? body : "");
    } else if (strcmp(method, "DELETE") == 0) {
        curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, "DELETE");
    }

    if (headers) curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

    CURLcode res = curl_easy_perform(curl);
    if (headers) curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        fprintf(stderr, "curl error: %s\n", curl_easy_strerror(res));
        return -1;
    }
    return 0;
}

void binance_hmac_sha256(const char *key, size_t key_len,
                         const char *data, size_t data_len,
                         char *out_hex) {
    unsigned char digest[EVP_MAX_MD_SIZE];
    unsigned int digest_len = 0;
    HMAC(EVP_sha256(), key, (int)key_len,
         (const unsigned char *)data, data_len, digest, &digest_len);
    for (unsigned int i = 0; i < digest_len; i++) {
        sprintf(out_hex + i * 2, "%02x", digest[i]);
    }
    out_hex[digest_len * 2] = '\0';
}

static void sign_query(const char *secret, const char *query, char *signed_query, size_t max_len) {
    char sig[65];
    binance_hmac_sha256(secret, strlen(secret), query, strlen(query), sig);
    snprintf(signed_query, max_len, "%s&signature=%s", query, sig);
}

static int check_api_error(cJSON *root, const char *func_name) {
    cJSON *err = cJSON_GetObjectItemCaseSensitive(root, "code");
    if (err && cJSON_IsNumber(err)) {
        cJSON *msg = cJSON_GetObjectItemCaseSensitive(root, "msg");
        fprintf(stderr, "%s: API error %d: %s\n",
                func_name, err->valueint, msg ? msg->valuestring : "unknown");
        return -1;
    }
    return 0;
}

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

int binance_get_klines(BinanceClient *client, const char *symbol, const char *interval,
                       int limit, Kline *out_klines, int *out_count) {
    *out_count = 0;
    if (limit > BINANCE_MAX_KLINES) limit = BINANCE_MAX_KLINES;

    char url[512];
    snprintf(url, sizeof(url), "%s/api/v3/klines?symbol=%s&interval=%s&limit=%d",
             client->base_url, symbol, interval, limit);

    char *response = malloc(BINANCE_MAX_RESPONSE);
    if (!response) return -1;
    response[0] = '\0';

    int rc = http_request(url, "GET", NULL, NULL, client->verbose, response, BINANCE_MAX_RESPONSE);
    if (rc != 0) { free(response); return -1; }

    cJSON *root = cJSON_Parse(response);
    free(response);
    if (!root) {
        fprintf(stderr, "binance_get_klines: JSON parse error\n");
        return -1;
    }
    if (!cJSON_IsArray(root)) {
        cJSON_Delete(root);
        fprintf(stderr, "binance_get_klines: expected array\n");
        return -1;
    }

    int n = cJSON_GetArraySize(root);
    if (n > limit) n = limit;
    for (int i = 0; i < n; i++) {
        cJSON *row = cJSON_GetArrayItem(root, i);
        if (!cJSON_IsArray(row) || cJSON_GetArraySize(row) < 7) continue;
        Kline *k = &out_klines[*out_count];
        k->open_time = (int64_t)cJSON_GetArrayItem(row, 0)->valuedouble;
        k->open = atof(cJSON_GetArrayItem(row, 1)->valuestring);
        k->high = atof(cJSON_GetArrayItem(row, 2)->valuestring);
        k->low = atof(cJSON_GetArrayItem(row, 3)->valuestring);
        k->close = atof(cJSON_GetArrayItem(row, 4)->valuestring);
        k->volume = atof(cJSON_GetArrayItem(row, 5)->valuestring);
        k->close_time = (int64_t)cJSON_GetArrayItem(row, 6)->valuedouble;
        (*out_count)++;
    }
    cJSON_Delete(root);
    return 0;
}

double binance_get_price(BinanceClient *client, const char *symbol) {
    char url[512];
    snprintf(url, sizeof(url), "%s/api/v3/ticker/price?symbol=%s", client->base_url, symbol);

    char response[4096];
    response[0] = '\0';
    if (http_request(url, "GET", NULL, NULL, client->verbose, response, sizeof(response)) != 0)
        return 0.0;

    cJSON *root = cJSON_Parse(response);
    if (!root) return 0.0;

    cJSON *price = cJSON_GetObjectItemCaseSensitive(root, "price");
    double val = 0.0;
    if (price && cJSON_IsString(price)) val = atof(price->valuestring);
    cJSON_Delete(root);
    return val;
}

int binance_place_order(BinanceClient *client, const char *symbol, OrderSide side,
                        OrderType type, double price, double qty,
                        OrderResponse *out_resp) {
    memset(out_resp, 0, sizeof(*out_resp));

    const char *side_str = (side == ORDER_BUY) ? "BUY" : "SELL";
    const char *type_str = (type == ORDER_LIMIT) ? "LIMIT" : "MARKET";

    char query[1024];
    int64_t ts = get_timestamp_ms();

    if (type == ORDER_LIMIT) {
        snprintf(query, sizeof(query),
                 "symbol=%s&side=%s&type=%s&timeInForce=GTC&price=%.8f&quantity=%.8f"
                 "&recvWindow=%d&timestamp=%lld",
                 symbol, side_str, type_str, price, qty,
                 client->recv_window, (long long)ts);
    } else {
        snprintf(query, sizeof(query),
                 "symbol=%s&side=%s&type=%s&quantity=%.8f&recvWindow=%d&timestamp=%lld",
                 symbol, side_str, type_str, qty,
                 client->recv_window, (long long)ts);
    }

    char signed_query[2048];
    sign_query(client->secret_key, query, signed_query, sizeof(signed_query));

    char url[2560];
    snprintf(url, sizeof(url), "%s/api/v3/order?%s", client->base_url, signed_query);

    char response[8192];
    response[0] = '\0';

    if (http_request(url, "POST", "", client->api_key, client->verbose,
                     response, sizeof(response)) != 0)
        return -1;

    cJSON *root = cJSON_Parse(response);
    if (!root) {
        fprintf(stderr, "binance_place_order: JSON parse error\n");
        return -1;
    }

    if (check_api_error(root, "binance_place_order") != 0) {
        cJSON_Delete(root);
        return -1;
    }

    cJSON *oid = cJSON_GetObjectItemCaseSensitive(root, "orderId");
    if (oid) out_resp->order_id = (int64_t)oid->valuedouble;

    cJSON *sym = cJSON_GetObjectItemCaseSensitive(root, "symbol");
    if (sym && cJSON_IsString(sym))
        strncpy(out_resp->symbol, sym->valuestring, BINANCE_MAX_SYMBOL - 1);

    cJSON *st = cJSON_GetObjectItemCaseSensitive(root, "status");
    if (st && cJSON_IsString(st))
        strncpy(out_resp->status, st->valuestring, sizeof(out_resp->status) - 1);

    out_resp->side = side;
    out_resp->type = type;
    out_resp->price = price;

    cJSON *oq = cJSON_GetObjectItemCaseSensitive(root, "origQty");
    if (oq && cJSON_IsString(oq)) out_resp->orig_qty = atof(oq->valuestring);

    cJSON *eq = cJSON_GetObjectItemCaseSensitive(root, "executedQty");
    if (eq && cJSON_IsString(eq)) out_resp->executed_qty = atof(eq->valuestring);

    cJSON_Delete(root);
    return 0;
}

int binance_cancel_order(BinanceClient *client, const char *symbol, int64_t order_id) {
    int64_t ts = get_timestamp_ms();
    char query[512];
    snprintf(query, sizeof(query),
             "symbol=%s&orderId=%lld&recvWindow=%d&timestamp=%lld",
             symbol, (long long)order_id, client->recv_window, (long long)ts);

    char signed_query[1024];
    sign_query(client->secret_key, query, signed_query, sizeof(signed_query));

    char url[2048];
    snprintf(url, sizeof(url), "%s/api/v3/order?%s", client->base_url, signed_query);

    char response[4096];
    response[0] = '\0';

    if (http_request(url, "DELETE", NULL, client->api_key, client->verbose,
                     response, sizeof(response)) != 0)
        return -1;

    cJSON *root = cJSON_Parse(response);
    if (!root) return -1;

    if (check_api_error(root, "binance_cancel_order") != 0) {
        cJSON_Delete(root);
        return -1;
    }
    cJSON_Delete(root);
    return 0;
}

int binance_get_margin_account(BinanceClient *client, MarginAsset *assets,
                               int max_assets, int *out_count, double *total_net_btc) {
    *out_count = 0;
    *total_net_btc = 0.0;

    int64_t ts = get_timestamp_ms();
    char query[256];
    snprintf(query, sizeof(query), "recvWindow=%d&timestamp=%lld",
             client->recv_window, (long long)ts);

    char signed_query[512];
    sign_query(client->secret_key, query, signed_query, sizeof(signed_query));

    char url[1024];
    snprintf(url, sizeof(url), "%s/sapi/v1/margin/account?%s", client->base_url, signed_query);

    char *response = malloc(BINANCE_MAX_RESPONSE);
    if (!response) return -1;
    response[0] = '\0';

    int rc = http_request(url, "GET", NULL, client->api_key, client->verbose,
                          response, BINANCE_MAX_RESPONSE);
    if (rc != 0) { free(response); return -1; }

    cJSON *root = cJSON_Parse(response);
    free(response);
    if (!root) return -1;

    if (check_api_error(root, "binance_get_margin_account") != 0) {
        cJSON_Delete(root);
        return -1;
    }

    cJSON *tna = cJSON_GetObjectItemCaseSensitive(root, "totalNetAssetOfBtc");
    if (tna && cJSON_IsString(tna)) *total_net_btc = atof(tna->valuestring);

    cJSON *user_assets = cJSON_GetObjectItemCaseSensitive(root, "userAssets");
    if (user_assets && cJSON_IsArray(user_assets)) {
        int n = cJSON_GetArraySize(user_assets);
        for (int i = 0; i < n && *out_count < max_assets; i++) {
            cJSON *item = cJSON_GetArrayItem(user_assets, i);

            cJSON *free_j = cJSON_GetObjectItemCaseSensitive(item, "free");
            cJSON *locked_j = cJSON_GetObjectItemCaseSensitive(item, "locked");
            cJSON *borrowed_j = cJSON_GetObjectItemCaseSensitive(item, "borrowed");
            cJSON *net_j = cJSON_GetObjectItemCaseSensitive(item, "netAsset");

            double free_v = (free_j && cJSON_IsString(free_j)) ? atof(free_j->valuestring) : 0.0;
            double locked_v = (locked_j && cJSON_IsString(locked_j)) ? atof(locked_j->valuestring) : 0.0;
            double borrowed_v = (borrowed_j && cJSON_IsString(borrowed_j)) ? atof(borrowed_j->valuestring) : 0.0;
            double net_v = (net_j && cJSON_IsString(net_j)) ? atof(net_j->valuestring) : 0.0;

            if (free_v == 0.0 && locked_v == 0.0 && borrowed_v == 0.0) continue;

            MarginAsset *a = &assets[*out_count];
            memset(a, 0, sizeof(*a));
            cJSON *asset_name = cJSON_GetObjectItemCaseSensitive(item, "asset");
            if (asset_name && cJSON_IsString(asset_name))
                strncpy(a->symbol, asset_name->valuestring, BINANCE_MAX_SYMBOL - 1);
            a->free_qty = free_v;
            a->locked_qty = locked_v;
            a->borrowed_qty = borrowed_v;
            a->net_asset = net_v;
            (*out_count)++;
        }
    }

    cJSON_Delete(root);
    return 0;
}
