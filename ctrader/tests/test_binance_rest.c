#include <stdio.h>
#include <string.h>
#include "../binance_rest.h"

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

static void test_init_cleanup(void) {
    BinanceClient client;
    int rc = binance_init(&client, "test_key", "test_secret");
    ASSERT_EQ_INT(rc, 0, "init: return code");
    ASSERT_STR_EQ(client.api_key, "test_key", "init: api_key");
    ASSERT_STR_EQ(client.base_url, "https://api.binance.com", "init: base_url");

    binance_cleanup(&client);
    ASSERT_EQ_INT(client.api_key[0], 0, "cleanup: key zeroed");
}

static void test_testnet(void) {
    BinanceClient client;
    binance_init(&client, "k", "s");
    binance_set_testnet(&client);
    ASSERT_STR_EQ(client.base_url, "https://testnet.binance.vision", "testnet: url");
    binance_cleanup(&client);
}

static void test_stubs_return_error(void) {
    BinanceClient client;
    binance_init(&client, "k", "s");

    Kline klines[10];
    int count = 0;
    int rc = binance_get_klines(&client, "BTCUSDT", "1h", 10, klines, &count);
    ASSERT_EQ_INT(rc, -1, "klines: stub returns -1");
    ASSERT_EQ_INT(count, 0, "klines: count=0");

    OrderResponse resp;
    rc = binance_place_order(&client, "BTCUSDT", ORDER_BUY, ORDER_LIMIT, 50000.0, 0.001, &resp);
    ASSERT_EQ_INT(rc, -1, "order: stub returns -1");

    rc = binance_cancel_order(&client, "BTCUSDT", 12345);
    ASSERT_EQ_INT(rc, -1, "cancel: stub returns -1");

    MarginAsset assets[5];
    int acount = 0;
    double net = 0;
    rc = binance_get_margin_account(&client, assets, 5, &acount, &net);
    ASSERT_EQ_INT(rc, -1, "margin: stub returns -1");

    binance_cleanup(&client);
}

static void test_hmac_stub(void) {
    char hex[65];
    binance_hmac_sha256("key", 3, "data", 4, hex);
    ASSERT_EQ_INT((int)strlen(hex), 64, "hmac: output length");
}

int main(void) {
    fprintf(stderr, "=== binance_rest tests ===\n");

    test_init_cleanup();
    test_testnet();
    test_stubs_return_error();
    test_hmac_stub();

    fprintf(stderr, "\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
