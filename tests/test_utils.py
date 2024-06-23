import time
import pytest

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
