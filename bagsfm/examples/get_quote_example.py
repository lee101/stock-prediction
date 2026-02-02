#!/usr/bin/env python3
"""
Example: Get a swap quote from Bags.fm API.

This shows how to:
1. Get a quote for swapping SOL -> token
2. Inspect the quote details (price impact, fees, etc.)
3. Estimate the SOL transaction costs

Usage:
    python -m bagsfm.examples.get_quote_example
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bagsfm import BagsAPIClient, BagsConfig
from bagsfm.config import SOL_MINT, CODEX_MINT
from bagsfm.utils import lamports_to_sol, format_sol, format_pct


async def main():
    """Get a quote for swapping SOL to CODEX."""
    # Create client
    config = BagsConfig()
    client = BagsAPIClient(config)

    # Amount to swap: 0.1 SOL in lamports
    amount_sol = 0.1
    amount_lamports = int(amount_sol * 1e9)

    print(f"\n{'='*60}")
    print(f"Getting quote: {amount_sol} SOL -> CODEX")
    print(f"{'='*60}\n")

    try:
        # Get quote
        quote = await client.get_quote(
            input_mint=SOL_MINT,
            output_mint=CODEX_MINT,
            amount=amount_lamports,
            slippage_bps=100,  # 1% slippage tolerance
        )

        # Display results
        print("Quote received:")
        print(f"  Input: {amount_sol} SOL ({amount_lamports} lamports)")
        print(f"  Output: {quote.out_amount} CODEX units")
        print(f"  Min output: {quote.min_out_amount} (after slippage)")
        print(f"  Price impact: {format_pct(quote.price_impact_pct / 100)}")
        print(f"  Effective rate: {quote.effective_rate:.8f} CODEX/lamport")

        # Fee information
        print(f"\nFee information:")
        print(f"  Slippage tolerance: {quote.slippage_bps} bps ({format_pct(quote.slippage_bps / 10000)})")
        print(f"  Estimated total fees: {quote.total_fee_estimate_bps:.2f} bps")

        if quote.platform_fee:
            print(f"  Platform fee: {quote.platform_fee}")

        if quote.out_transfer_fee:
            print(f"  Token-2022 transfer fee: {quote.out_transfer_fee}")

        # Route information
        if quote.route_plan:
            print(f"\nRoute ({len(quote.route_plan)} leg(s)):")
            for i, leg in enumerate(quote.route_plan, 1):
                print(f"  Leg {i}: {leg.source}")
                print(f"    {leg.amount_in} -> {leg.amount_out}")
                if leg.fee_amount > 0:
                    print(f"    Fee: {leg.fee_amount}")

        # Compute estimates
        if quote.simulated_compute_units:
            print(f"\nCompute estimate: {quote.simulated_compute_units} units")

        # Now let's see what the transaction would cost
        print(f"\n{'='*60}")
        print("Building swap transaction (to see fees)...")
        print(f"{'='*60}\n")

        # Note: This requires a valid public key
        # For this example, we'll use a dummy key
        dummy_pubkey = "11111111111111111111111111111111"

        try:
            swap_tx = await client.build_swap_transaction(
                quote=quote,
                user_public_key=dummy_pubkey,
            )

            print("Transaction built:")
            print(f"  Compute limit: {swap_tx.compute_unit_limit}")
            print(f"  Priority fee: {swap_tx.prioritization_fee_lamports} lamports ({lamports_to_sol(swap_tx.prioritization_fee_lamports):.9f} SOL)")
            print(f"  Estimated total SOL fee: {format_sol(swap_tx.estimated_sol_fee)}")
            print(f"  Last valid block: {swap_tx.last_valid_block_height}")

        except Exception as e:
            print(f"Note: Could not build transaction (expected with dummy key): {e}")

    except Exception as e:
        print(f"Error getting quote: {e}")
        print("\nMake sure BAGS_API_KEY is set in your environment or env_real.py")


if __name__ == "__main__":
    asyncio.run(main())
