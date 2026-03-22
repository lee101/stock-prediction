#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "../binance_rest.h"
#include "../vendor/cJSON.h"

static int g_pass = 0, g_fail = 0;

#define ASSERT_EQ_INT(a, b, msg) do { \
    int _a = (a), _b = (b); \
    if (_a != _b) { \
        fprintf(stderr, "FAIL %s: %d != %d\n", msg, _a, _b); \
        g_fail++; \
    } else { \
        g_pass++; \
    } \
} while(0)

#define ASSERT_STR_EQ(a, b, msg) do { \
    if (strcmp((a), (b)) != 0) { \
        fprintf(stderr, "FAIL %s: '%s' != '%s'\n", msg, (a), (b)); \
        g_fail++; \
    } else { \
        g_pass++; \
    } \
} while(0)

#define ASSERT_NEAR(a, b, eps, msg) do { \
    double _a = (a), _b = (b); \
    if (fabs(_a - _b) > (eps)) { \
        fprintf(stderr, "FAIL %s: %.10f != %.10f\n", msg, _a, _b); \
        g_fail++; \
    } else { \
        g_pass++; \
    } \
} while(0)

#define ASSERT_TRUE(cond, msg) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s\n", msg); \
        g_fail++; \
    } else { \
        g_pass++; \
    } \
} while(0)

static void test_init_cleanup(void) {
    BinanceClient client;
    int rc = binance_init(&client, "test_key", "test_secret");
    ASSERT_EQ_INT(rc, 0, "init: return code");
    ASSERT_STR_EQ(client.api_key, "test_key", "init: api_key");
    ASSERT_STR_EQ(client.secret_key, "test_secret", "init: secret_key");
    ASSERT_STR_EQ(client.base_url, "https://api.binance.com", "init: base_url");
    ASSERT_EQ_INT(client.recv_window, 5000, "init: recv_window");

    binance_cleanup(&client);
    ASSERT_EQ_INT(client.api_key[0], 0, "cleanup: key zeroed");
    ASSERT_EQ_INT(client.secret_key[0], 0, "cleanup: secret zeroed");
}

static void test_init_null_keys(void) {
    BinanceClient client;
    int rc = binance_init(&client, NULL, NULL);
    ASSERT_EQ_INT(rc, 0, "init_null: return code");
    ASSERT_EQ_INT(client.api_key[0], 0, "init_null: api_key empty");
    ASSERT_EQ_INT(client.secret_key[0], 0, "init_null: secret_key empty");
    binance_cleanup(&client);
}

static void test_testnet(void) {
    BinanceClient client;
    binance_init(&client, "k", "s");
    binance_set_testnet(&client);
    ASSERT_STR_EQ(client.base_url, "https://testnet.binance.vision", "testnet: url");
    binance_cleanup(&client);
}

static void test_hmac_sha256(void) {
    char hex[65];
    /* Test vector: HMAC-SHA256("key", "The quick brown fox jumps over the lazy dog")
       = f7bc83f430538424b13298e6aa6fb143ef4d59a14946175997479dbc2d1a3cd8 */
    binance_hmac_sha256("key", 3,
                        "The quick brown fox jumps over the lazy dog", 43,
                        hex);
    ASSERT_STR_EQ(hex, "f7bc83f430538424b13298e6aa6fb143ef4d59a14946175997479dbc2d1a3cd8",
                  "hmac: known vector");
    ASSERT_EQ_INT((int)strlen(hex), 64, "hmac: output length");
}

static void test_hmac_sha256_empty(void) {
    char hex[65];
    /* HMAC-SHA256("", "") =
       b613679a0814d9ec772f95d778c35fc5ff1697c493715653c6c712144292c5ad */
    binance_hmac_sha256("", 0, "", 0, hex);
    ASSERT_STR_EQ(hex, "b613679a0814d9ec772f95d778c35fc5ff1697c493715653c6c712144292c5ad",
                  "hmac_empty: known vector");
}

static void test_hmac_sha256_binance_example(void) {
    char hex[65];
    /* Binance docs example:
       key = "NhqPtmdSJYdKjVHjA7PZj4Mge3R5YNiP1e3UZjInClVN65XAbvqqM6A7H5fATj0j"
       data = "symbol=LTCBTC&side=BUY&type=LIMIT&timeInForce=GTC&quantity=1&price=0.1&recvWindow=5000&timestamp=1499827319559"
       sig = "c8db56825ae71d6d79447849e617115f4a920fa2acdcab2b053c4b2838bd6b71" */
    const char *key = "NhqPtmdSJYdKjVHjA7PZj4Mge3R5YNiP1e3UZjInClVN65XAbvqqM6A7H5fATj0j";
    const char *data = "symbol=LTCBTC&side=BUY&type=LIMIT&timeInForce=GTC&quantity=1&price=0.1&recvWindow=5000&timestamp=1499827319559";
    binance_hmac_sha256(key, strlen(key), data, strlen(data), hex);
    ASSERT_STR_EQ(hex, "c8db56825ae71d6d79447849e617115f4a920fa2acdcab2b053c4b2838bd6b71",
                  "hmac_binance: official example");
}

static void test_parse_klines_json(void) {
    const char *json =
        "["
        "  [1609459200000, \"29000.00\", \"29500.00\", \"28800.00\", \"29200.00\", \"1234.56\", 1609462799999],"
        "  [1609462800000, \"29200.00\", \"29600.00\", \"29000.00\", \"29400.00\", \"987.65\", 1609466399999],"
        "  [1609466400000, \"29400.00\", \"30000.00\", \"29300.00\", \"29900.00\", \"2345.67\", 1609469999999]"
        "]";

    cJSON *root = cJSON_Parse(json);
    ASSERT_TRUE(root != NULL, "klines_parse: root not null");
    ASSERT_TRUE(cJSON_IsArray(root), "klines_parse: is array");
    ASSERT_EQ_INT(cJSON_GetArraySize(root), 3, "klines_parse: array size");

    Kline klines[3];
    int count = 0;
    int n = cJSON_GetArraySize(root);
    for (int i = 0; i < n; i++) {
        cJSON *row = cJSON_GetArrayItem(root, i);
        if (!cJSON_IsArray(row) || cJSON_GetArraySize(row) < 7) continue;
        Kline *k = &klines[count];
        k->open_time = (int64_t)cJSON_GetArrayItem(row, 0)->valuedouble;
        k->open = atof(cJSON_GetArrayItem(row, 1)->valuestring);
        k->high = atof(cJSON_GetArrayItem(row, 2)->valuestring);
        k->low = atof(cJSON_GetArrayItem(row, 3)->valuestring);
        k->close = atof(cJSON_GetArrayItem(row, 4)->valuestring);
        k->volume = atof(cJSON_GetArrayItem(row, 5)->valuestring);
        k->close_time = (int64_t)cJSON_GetArrayItem(row, 6)->valuedouble;
        count++;
    }

    ASSERT_EQ_INT(count, 3, "klines_parse: parsed 3");
    ASSERT_NEAR(klines[0].open, 29000.0, 0.01, "klines[0].open");
    ASSERT_NEAR(klines[0].high, 29500.0, 0.01, "klines[0].high");
    ASSERT_NEAR(klines[0].low, 28800.0, 0.01, "klines[0].low");
    ASSERT_NEAR(klines[0].close, 29200.0, 0.01, "klines[0].close");
    ASSERT_NEAR(klines[0].volume, 1234.56, 0.01, "klines[0].volume");
    ASSERT_TRUE(klines[0].open_time == 1609459200000LL, "klines[0].open_time");
    ASSERT_TRUE(klines[0].close_time == 1609462799999LL, "klines[0].close_time");

    ASSERT_NEAR(klines[2].close, 29900.0, 0.01, "klines[2].close");
    ASSERT_NEAR(klines[2].volume, 2345.67, 0.01, "klines[2].volume");

    cJSON_Delete(root);
}

static void test_parse_order_response(void) {
    const char *json =
        "{"
        "  \"symbol\": \"BTCUSDT\","
        "  \"orderId\": 28,"
        "  \"clientOrderId\": \"6gCrw2kRUAF9CvJDGP16IP\","
        "  \"transactTime\": 1507725176595,"
        "  \"price\": \"0.00000000\","
        "  \"origQty\": \"10.00000000\","
        "  \"executedQty\": \"10.00000000\","
        "  \"status\": \"FILLED\","
        "  \"type\": \"MARKET\","
        "  \"side\": \"SELL\""
        "}";

    cJSON *root = cJSON_Parse(json);
    ASSERT_TRUE(root != NULL, "order_parse: root not null");

    cJSON *oid = cJSON_GetObjectItemCaseSensitive(root, "orderId");
    ASSERT_TRUE(oid != NULL, "order_parse: orderId exists");
    ASSERT_EQ_INT((int)oid->valuedouble, 28, "order_parse: orderId=28");

    cJSON *sym = cJSON_GetObjectItemCaseSensitive(root, "symbol");
    ASSERT_STR_EQ(sym->valuestring, "BTCUSDT", "order_parse: symbol");

    cJSON *st = cJSON_GetObjectItemCaseSensitive(root, "status");
    ASSERT_STR_EQ(st->valuestring, "FILLED", "order_parse: status");

    cJSON *oq = cJSON_GetObjectItemCaseSensitive(root, "origQty");
    ASSERT_NEAR(atof(oq->valuestring), 10.0, 0.001, "order_parse: origQty");

    cJSON *eq = cJSON_GetObjectItemCaseSensitive(root, "executedQty");
    ASSERT_NEAR(atof(eq->valuestring), 10.0, 0.001, "order_parse: executedQty");

    cJSON_Delete(root);
}

static void test_parse_margin_account(void) {
    const char *json =
        "{"
        "  \"totalNetAssetOfBtc\": \"1.23456789\","
        "  \"userAssets\": ["
        "    {"
        "      \"asset\": \"BTC\","
        "      \"free\": \"0.5\","
        "      \"locked\": \"0.1\","
        "      \"borrowed\": \"0.0\","
        "      \"netAsset\": \"0.6\""
        "    },"
        "    {"
        "      \"asset\": \"USDT\","
        "      \"free\": \"1000.0\","
        "      \"locked\": \"0.0\","
        "      \"borrowed\": \"500.0\","
        "      \"netAsset\": \"500.0\""
        "    },"
        "    {"
        "      \"asset\": \"ETH\","
        "      \"free\": \"0.0\","
        "      \"locked\": \"0.0\","
        "      \"borrowed\": \"0.0\","
        "      \"netAsset\": \"0.0\""
        "    }"
        "  ]"
        "}";

    cJSON *root = cJSON_Parse(json);
    ASSERT_TRUE(root != NULL, "margin_parse: root not null");

    cJSON *tna = cJSON_GetObjectItemCaseSensitive(root, "totalNetAssetOfBtc");
    ASSERT_TRUE(tna != NULL, "margin_parse: totalNetAssetOfBtc");
    ASSERT_NEAR(atof(tna->valuestring), 1.23456789, 1e-8, "margin_parse: net_btc");

    cJSON *ua = cJSON_GetObjectItemCaseSensitive(root, "userAssets");
    ASSERT_TRUE(cJSON_IsArray(ua), "margin_parse: userAssets is array");
    ASSERT_EQ_INT(cJSON_GetArraySize(ua), 3, "margin_parse: 3 assets");

    /* Simulate filtering: skip zero-balance assets */
    MarginAsset assets[10];
    int count = 0;
    int n = cJSON_GetArraySize(ua);
    for (int i = 0; i < n; i++) {
        cJSON *item = cJSON_GetArrayItem(ua, i);
        cJSON *free_j = cJSON_GetObjectItemCaseSensitive(item, "free");
        cJSON *locked_j = cJSON_GetObjectItemCaseSensitive(item, "locked");
        cJSON *borrowed_j = cJSON_GetObjectItemCaseSensitive(item, "borrowed");
        double fv = atof(free_j->valuestring);
        double lv = atof(locked_j->valuestring);
        double bv = atof(borrowed_j->valuestring);
        if (fv == 0.0 && lv == 0.0 && bv == 0.0) continue;
        cJSON *asset_name = cJSON_GetObjectItemCaseSensitive(item, "asset");
        strncpy(assets[count].symbol, asset_name->valuestring, BINANCE_MAX_SYMBOL - 1);
        assets[count].free_qty = fv;
        assets[count].locked_qty = lv;
        assets[count].borrowed_qty = bv;
        cJSON *net_j = cJSON_GetObjectItemCaseSensitive(item, "netAsset");
        assets[count].net_asset = atof(net_j->valuestring);
        count++;
    }

    ASSERT_EQ_INT(count, 2, "margin_filter: 2 non-zero assets");
    ASSERT_STR_EQ(assets[0].symbol, "BTC", "margin_filter: first=BTC");
    ASSERT_NEAR(assets[0].free_qty, 0.5, 0.001, "margin_filter: BTC free");
    ASSERT_NEAR(assets[0].locked_qty, 0.1, 0.001, "margin_filter: BTC locked");
    ASSERT_NEAR(assets[0].net_asset, 0.6, 0.001, "margin_filter: BTC net");
    ASSERT_STR_EQ(assets[1].symbol, "USDT", "margin_filter: second=USDT");
    ASSERT_NEAR(assets[1].borrowed_qty, 500.0, 0.001, "margin_filter: USDT borrowed");
    ASSERT_NEAR(assets[1].net_asset, 500.0, 0.001, "margin_filter: USDT net");

    cJSON_Delete(root);
}

static void test_parse_error_response(void) {
    const char *json = "{\"code\": -1021, \"msg\": \"Timestamp for this request was 1000ms ahead of the server's time.\"}";
    cJSON *root = cJSON_Parse(json);
    ASSERT_TRUE(root != NULL, "error_parse: root");
    cJSON *code = cJSON_GetObjectItemCaseSensitive(root, "code");
    ASSERT_TRUE(code != NULL, "error_parse: code exists");
    ASSERT_EQ_INT(code->valueint, -1021, "error_parse: code=-1021");
    cJSON *msg = cJSON_GetObjectItemCaseSensitive(root, "msg");
    ASSERT_TRUE(msg != NULL, "error_parse: msg exists");
    ASSERT_TRUE(strstr(msg->valuestring, "Timestamp") != NULL, "error_parse: msg content");
    cJSON_Delete(root);
}

static void test_parse_price_response(void) {
    const char *json = "{\"symbol\": \"BTCUSDT\", \"price\": \"67543.21000000\"}";
    cJSON *root = cJSON_Parse(json);
    ASSERT_TRUE(root != NULL, "price_parse: root");
    cJSON *price = cJSON_GetObjectItemCaseSensitive(root, "price");
    ASSERT_TRUE(price != NULL, "price_parse: price exists");
    ASSERT_NEAR(atof(price->valuestring), 67543.21, 0.01, "price_parse: value");
    cJSON_Delete(root);
}

static void test_parse_klines_empty(void) {
    const char *json = "[]";
    cJSON *root = cJSON_Parse(json);
    ASSERT_TRUE(root != NULL, "klines_empty: root");
    ASSERT_TRUE(cJSON_IsArray(root), "klines_empty: is array");
    ASSERT_EQ_INT(cJSON_GetArraySize(root), 0, "klines_empty: size=0");
    cJSON_Delete(root);
}

static void test_parse_klines_malformed_row(void) {
    /* Row with only 5 elements should be skipped */
    const char *json =
        "["
        "  [1609459200000, \"29000.00\", \"29500.00\", \"28800.00\", \"29200.00\"],"
        "  [1609462800000, \"29200.00\", \"29600.00\", \"29000.00\", \"29400.00\", \"987.65\", 1609466399999]"
        "]";
    cJSON *root = cJSON_Parse(json);
    ASSERT_TRUE(root != NULL, "klines_malformed: root");
    int count = 0;
    int n = cJSON_GetArraySize(root);
    for (int i = 0; i < n; i++) {
        cJSON *row = cJSON_GetArrayItem(root, i);
        if (!cJSON_IsArray(row) || cJSON_GetArraySize(row) < 7) continue;
        count++;
    }
    ASSERT_EQ_INT(count, 1, "klines_malformed: only 1 valid row");
    cJSON_Delete(root);
}

int main(void) {
    fprintf(stderr, "=== binance_rest tests ===\n");

    test_init_cleanup();
    test_init_null_keys();
    test_testnet();
    test_hmac_sha256();
    test_hmac_sha256_empty();
    test_hmac_sha256_binance_example();
    test_parse_klines_json();
    test_parse_order_response();
    test_parse_margin_account();
    test_parse_error_response();
    test_parse_price_response();
    test_parse_klines_empty();
    test_parse_klines_malformed_row();

    fprintf(stderr, "\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
