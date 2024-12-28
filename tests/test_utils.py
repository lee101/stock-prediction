import time

from src.utils import debounce

call_count = 0


@debounce(2)  # 2 seconds debounce period
def debounced_function():
    global call_count
    call_count += 1


def test_debounce():
    global call_count

    # Call the function twice in quick succession
    debounced_function()
    debounced_function()

    # Assert that the function was only called once due to debounce
    assert call_count == 1

    # Wait for the debounce period to pass
    time.sleep(2)

    # Call the function again
    debounced_function()
    debounced_function()

    # Assert that the function was called again after debounce period
    assert call_count == 2


@debounce(2, key_func=lambda x: x)
def debounced_function_with_key(x):
    global call_count
    call_count += 1


def test_debounce_with_key():
    global call_count
    call_count = 0

    # Call the function with different keys
    debounced_function_with_key(1)
    debounced_function_with_key(2)
    debounced_function_with_key(1)

    # Assert that the function was called twice (once for each unique key)
    assert call_count == 2

    # Wait for the debounce period to pass
    time.sleep(2)

    # Call the function again with the same keys
    debounced_function_with_key(1)
    debounced_function_with_key(2)

    # Assert that the function was called two more times after debounce period
    assert call_count == 4

    # Call the function immediately with the same keys
    debounced_function_with_key(1)
    debounced_function_with_key(2)

    # Assert that the call count hasn't changed due to debounce
    assert call_count == 4
