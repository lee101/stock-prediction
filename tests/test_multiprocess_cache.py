"""
Multi-process test to verify disk cache is shared across processes.

This test spawns multiple subprocesses to verify that the disk cache
works correctly when multiple scripts run simultaneously.
"""

import os
import sys
import subprocess
import time
from pathlib import Path


def test_multiprocess_cache_with_paper():
    """Test that cache is shared across multiple processes with PAPER=1."""
    # Set PAPER=1 for this test
    env = os.environ.copy()
    env['PAPER'] = '1'

    # Script to run in subprocess
    test_script = """
import os
import sys
sys.path.insert(0, '/nvme0n1-disk/code/stock-prediction')
from unittest.mock import patch, MagicMock
from types import SimpleNamespace
import alpaca_wrapper
import time

# Mock the alpaca_api to avoid actual API calls
mock_account = SimpleNamespace(
    equity='100000.00',
    cash='20000.00',
    multiplier='1.0',
    buying_power='100000.00',
)

# Clear cache first
cache_key = f"account_{alpaca_wrapper._IS_PAPER}"
alpaca_wrapper._account_diskcache.delete(cache_key)

# Patch and set the account in cache
with patch('alpaca_wrapper.alpaca_api') as mock_api:
    mock_api.get_account.return_value = mock_account

    # First call - populates cache
    account = alpaca_wrapper.get_account(use_cache=True)
    print(f"Process {os.getpid()}: First call made API request")

    # Wait a bit
    time.sleep(0.5)

    # Second call - should use cache
    account2 = alpaca_wrapper.get_account(use_cache=True)
    print(f"Process {os.getpid()}: Second call used cache (API call count: {mock_api.get_account.call_count})")

    # Verify only 1 API call was made
    assert mock_api.get_account.call_count == 1, f"Expected 1 API call, got {mock_api.get_account.call_count}"
    print(f"Process {os.getpid()}: SUCCESS - Cache working correctly")
"""

    # Run multiple processes
    processes = []
    for i in range(3):
        p = subprocess.Popen(
            [sys.executable, '-c', test_script],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        processes.append(p)
        time.sleep(0.1)  # Stagger the starts slightly

    # Wait for all processes and collect results
    results = []
    for p in processes:
        stdout, stderr = p.communicate(timeout=10)
        results.append({
            'returncode': p.returncode,
            'stdout': stdout,
            'stderr': stderr,
        })

    # Check all processes succeeded
    for i, result in enumerate(results):
        print(f"\n=== Process {i} ===")
        print(f"Return code: {result['returncode']}")
        print(f"STDOUT:\n{result['stdout']}")
        if result['stderr']:
            print(f"STDERR:\n{result['stderr']}")

        assert result['returncode'] == 0, f"Process {i} failed with code {result['returncode']}"
        assert 'SUCCESS' in result['stdout'], f"Process {i} did not complete successfully"

    print("\n✓ All processes completed successfully")
    print("✓ Disk cache is working across multiple processes")


def test_clock_cache_with_paper():
    """Test that clock cache works with PAPER=1."""
    env = os.environ.copy()
    env['PAPER'] = '1'

    test_script = """
import sys
sys.path.insert(0, '/nvme0n1-disk/code/stock-prediction')
from unittest.mock import patch
from types import SimpleNamespace
import alpaca_wrapper

mock_clock = SimpleNamespace(
    is_open=True,
    timestamp='2025-11-18T10:00:00Z',
)

# Clear cache
cache_key = f"clock_{alpaca_wrapper._IS_PAPER}"
alpaca_wrapper._account_diskcache.delete(cache_key)

with patch('alpaca_wrapper.alpaca_api') as mock_api:
    mock_api.get_clock.return_value = mock_clock

    # First call
    clock1 = alpaca_wrapper.get_clock_internal(use_cache=True)

    # Second call - should use cache
    clock2 = alpaca_wrapper.get_clock_internal(use_cache=True)

    # Verify only 1 API call
    assert mock_api.get_clock.call_count == 1, f"Expected 1 API call, got {mock_api.get_clock.call_count}"
    print("SUCCESS - Clock cache working")
"""

    result = subprocess.run(
        [sys.executable, '-c', test_script],
        env=env,
        capture_output=True,
        text=True,
        timeout=10
    )

    print(f"Return code: {result.returncode}")
    print(f"STDOUT:\n{result.stdout}")
    if result.stderr:
        print(f"STDERR:\n{result.stderr}")

    assert result.returncode == 0, f"Clock cache test failed with code {result.returncode}"
    assert 'SUCCESS' in result.stdout
    print("✓ Clock cache test passed")


if __name__ == '__main__':
    print("Testing multi-process cache sharing with PAPER=1...\n")
    test_multiprocess_cache_with_paper()
    print("\n" + "="*60 + "\n")
    test_clock_cache_with_paper()
    print("\n✓ All multi-process tests passed!")
