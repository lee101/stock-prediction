"""Bags.fm API wrapper for Solana swaps.

Provides:
- Quote API for getting swap quotes
- Swap API for building transactions
- Transaction signing and sending
- Cost estimation and breakdown
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx

from .config import BagsConfig, SOL_MINT

logger = logging.getLogger(__name__)


@dataclass
class RouteLeg:
    """A single leg/hop in a swap route."""

    input_mint: str
    output_mint: str
    amount_in: int
    amount_out: int
    fee_amount: int
    fee_mint: str
    source: str  # AMM/DEX name


@dataclass
class QuoteResponse:
    """Response from Bags quote API."""

    input_mint: str
    output_mint: str
    in_amount: int  # Input amount in smallest units (lamports/etc)
    out_amount: int  # Expected output amount
    min_out_amount: int  # Minimum output after slippage
    price_impact_pct: float  # Price impact as percentage
    slippage_bps: int  # Slippage tolerance in basis points

    # Fee information
    platform_fee: Optional[Dict[str, Any]] = None
    out_transfer_fee: Optional[int] = None  # Token-2022 transfer fee

    # Route information
    route_plan: List[RouteLeg] = field(default_factory=list)

    # Compute hints
    simulated_compute_units: Optional[int] = None

    # Raw response for swap building
    raw_response: Dict[str, Any] = field(default_factory=dict)

    @property
    def effective_rate(self) -> float:
        """Effective exchange rate (out/in)."""
        if self.in_amount <= 0:
            return 0.0
        return self.out_amount / self.in_amount

    @property
    def total_fee_estimate_bps(self) -> float:
        """Estimated total fees in basis points."""
        # Price impact + any platform fees
        impact_bps = self.price_impact_pct * 100
        platform_bps = 0.0
        if self.platform_fee:
            platform_bps = float(self.platform_fee.get("feeBps", 0))
        return impact_bps + platform_bps


@dataclass
class SwapTransaction:
    """Response from Bags swap API with transaction details."""

    swap_transaction: str  # Base58 encoded serialized VersionedTransaction
    compute_unit_limit: int
    prioritization_fee_lamports: int
    last_valid_block_height: int

    # Additional metadata
    estimated_sol_fee: float = 0.0  # Estimated total SOL fee


@dataclass
class SwapResult:
    """Result of executing a swap."""

    success: bool
    signature: Optional[str] = None
    error: Optional[str] = None

    # Transaction costs (after confirmation)
    fee_lamports: Optional[int] = None
    payer_total_spent_lamports: Optional[int] = None

    # Token amounts
    input_amount: Optional[int] = None
    output_amount: Optional[int] = None

    # Timing
    timestamp: datetime = field(default_factory=datetime.utcnow)
    confirmation_time_ms: Optional[int] = None


class BagsAPIClient:
    """Client for Bags.fm API.

    Provides methods for:
    - Getting swap quotes
    - Building swap transactions
    - Signing and sending transactions
    - Getting price and fee information
    """

    def __init__(self, config: BagsConfig) -> None:
        self.config = config
        self._headers = {
            "x-api-key": config.api_key,
            "Content-Type": "application/json",
        }

    async def get_quote(
        self,
        input_mint: str,
        output_mint: str,
        amount: int,
        slippage_bps: int = 100,
        slippage_mode: str = "manual",
    ) -> QuoteResponse:
        """Get a swap quote from Bags API.

        Args:
            input_mint: Input token mint address
            output_mint: Output token mint address
            amount: Amount to swap in smallest units (lamports for SOL)
            slippage_bps: Slippage tolerance in basis points (100 = 1%)
            slippage_mode: "manual" or "auto"

        Returns:
            QuoteResponse with quote details

        Raises:
            RuntimeError: If quote request fails
        """
        params = {
            "inputMint": input_mint,
            "outputMint": output_mint,
            "amount": str(amount),
            "slippageMode": slippage_mode,
            "slippageBps": str(slippage_bps),
        }

        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            response = await client.get(
                f"{self.config.base_url}/trade/quote",
                headers=self._headers,
                params=params,
            )
            data = response.json()

            if not data.get("success"):
                error_msg = data.get("error", data.get("message", "Unknown error"))
                raise RuntimeError(f"Quote failed: {error_msg}")

            resp = data["response"]

            # Parse route legs
            route_plan = []
            for leg in resp.get("routePlan", []):
                route_plan.append(
                    RouteLeg(
                        input_mint=leg.get("inputMint", ""),
                        output_mint=leg.get("outputMint", ""),
                        amount_in=int(leg.get("inAmount", 0)),
                        amount_out=int(leg.get("outAmount", 0)),
                        fee_amount=int(leg.get("feeAmount", 0)),
                        fee_mint=leg.get("feeMint", ""),
                        source=leg.get("source", ""),
                    )
                )

            return QuoteResponse(
                input_mint=input_mint,
                output_mint=output_mint,
                in_amount=int(resp.get("inAmount", amount)),
                out_amount=int(resp.get("outAmount", 0)),
                min_out_amount=int(resp.get("minOutAmount", 0)),
                price_impact_pct=float(resp.get("priceImpactPct", 0)),
                slippage_bps=slippage_bps,
                platform_fee=resp.get("platformFee"),
                out_transfer_fee=resp.get("outTransferFee"),
                route_plan=route_plan,
                simulated_compute_units=resp.get("simulatedComputeUnits"),
                raw_response=resp,
            )

    async def build_swap_transaction(
        self,
        quote: QuoteResponse,
        user_public_key: str,
    ) -> SwapTransaction:
        """Build a swap transaction from a quote.

        Args:
            quote: Quote response from get_quote()
            user_public_key: Base58 encoded user public key

        Returns:
            SwapTransaction with transaction data

        Raises:
            RuntimeError: If swap transaction building fails
        """
        payload = {
            "quoteResponse": quote.raw_response,
            "userPublicKey": user_public_key,
        }

        async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
            response = await client.post(
                f"{self.config.base_url}/trade/swap",
                headers=self._headers,
                json=payload,
            )
            data = response.json()

            if not data.get("success"):
                error_msg = data.get("error", data.get("message", "Unknown error"))
                raise RuntimeError(f"Swap transaction build failed: {error_msg}")

            resp = data["response"]

            swap_tx = SwapTransaction(
                swap_transaction=resp["swapTransaction"],
                compute_unit_limit=int(resp.get("computeUnitLimit", 200000)),
                prioritization_fee_lamports=int(resp.get("prioritizationFeeLamports", 0)),
                last_valid_block_height=int(resp.get("lastValidBlockHeight", 0)),
            )

            # Estimate total SOL fee
            # Base fee (5000 lamports) + priority fee
            swap_tx.estimated_sol_fee = (5000 + swap_tx.prioritization_fee_lamports) / 1e9

            return swap_tx

    async def get_price_usd(self, mint: str) -> Optional[float]:
        """Get USD price for a token using Jupiter Price API.

        Args:
            mint: Token mint address

        Returns:
            USD price or None if unavailable
        """
        try:
            from env_real import JUP_API_KEY
        except ImportError:
            import os
            JUP_API_KEY = os.getenv("JUP_API_KEY", "")

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(
                    "https://api.jup.ag/price/v3",
                    headers={"x-api-key": JUP_API_KEY},
                    params={"ids": mint},
                )
                data = response.json()

                if data.get("data") and mint in data["data"]:
                    return float(data["data"][mint].get("price", 0))

        except Exception as e:
            logger.warning(f"Failed to get price for {mint}: {e}")

        return None

    async def get_sol_price_usd(self) -> float:
        """Get current SOL/USD price."""
        price = await self.get_price_usd(SOL_MINT)
        return price if price else 0.0


class SolanaTransactionExecutor:
    """Executes Solana transactions.

    Handles signing and sending transactions to the network.
    """

    def __init__(self, config: BagsConfig) -> None:
        self.config = config
        self._keypair = None

    def _get_keypair(self):
        """Lazy load keypair from config."""
        if self._keypair is None:
            if not self.config.private_key_b58:
                raise ValueError("No private key configured")

            try:
                from solders.keypair import Keypair
                import base58

                secret_bytes = base58.b58decode(self.config.private_key_b58)
                self._keypair = Keypair.from_bytes(secret_bytes)
            except ImportError:
                raise ImportError(
                    "solders and base58 packages required. "
                    "Install with: pip install solders base58"
                )

        return self._keypair

    @property
    def public_key(self) -> str:
        """Get the public key as base58 string."""
        return str(self._get_keypair().pubkey())

    async def sign_and_send(
        self,
        swap_tx: SwapTransaction,
        skip_preflight: bool = False,
    ) -> SwapResult:
        """Sign and send a swap transaction.

        Args:
            swap_tx: SwapTransaction from build_swap_transaction()
            skip_preflight: If True, skip preflight simulation

        Returns:
            SwapResult with execution details
        """
        try:
            from solana.rpc.async_api import AsyncClient
            from solana.rpc.types import TxOpts
            from solders.transaction import VersionedTransaction
            from solders.message import to_bytes_versioned
        except ImportError:
            raise ImportError(
                "solana and solders packages required. "
                "Install with: pip install solana solders"
            )

        start_time = datetime.utcnow()

        try:
            keypair = self._get_keypair()

            # Decode and deserialize transaction (Base58 encoded from Bags API)
            import base58
            tx_bytes = base58.b58decode(swap_tx.swap_transaction)
            vtx = VersionedTransaction.from_bytes(tx_bytes)

            # Sign the transaction
            msg_bytes = to_bytes_versioned(vtx.message)
            sig = keypair.sign_message(msg_bytes)

            # Find payer index and set signature
            account_keys = list(vtx.message.account_keys)
            payer_index = account_keys.index(keypair.pubkey())
            sigs = list(vtx.signatures)
            sigs[payer_index] = sig

            # Create new transaction with updated signatures using populate()
            vtx = VersionedTransaction.populate(vtx.message, sigs)

            # Send transaction
            async with AsyncClient(self.config.rpc_url) as rpc:
                opts = TxOpts(skip_preflight=skip_preflight, preflight_commitment="confirmed")
                result = await rpc.send_raw_transaction(bytes(vtx), opts=opts)

                if result.value is None:
                    return SwapResult(
                        success=False,
                        error=f"Transaction send failed: {result}",
                    )

                signature_obj = result.value  # Signature object
                signature = str(signature_obj)  # String for logging/return

                # Wait for confirmation
                await rpc.confirm_transaction(signature_obj, commitment="confirmed")

                # Get transaction details for cost breakdown
                tx_detail = await rpc.get_transaction(
                    signature_obj, max_supported_transaction_version=0
                )

                fee_lamports = None
                payer_spent = None

                if tx_detail.value:
                    meta = tx_detail.value.transaction.meta
                    fee_lamports = meta.fee

                    # Calculate payer total spent
                    pre_balances = meta.pre_balances
                    post_balances = meta.post_balances
                    if payer_index < len(pre_balances):
                        payer_spent = pre_balances[payer_index] - post_balances[payer_index]

                end_time = datetime.utcnow()
                confirmation_ms = int((end_time - start_time).total_seconds() * 1000)

                return SwapResult(
                    success=True,
                    signature=signature,
                    fee_lamports=fee_lamports,
                    payer_total_spent_lamports=payer_spent,
                    timestamp=start_time,
                    confirmation_time_ms=confirmation_ms,
                )

        except Exception as e:
            logger.exception(f"Transaction execution failed: {e}")
            return SwapResult(
                success=False,
                error=str(e),
                timestamp=start_time,
            )

    async def estimate_fee(
        self,
        swap_tx: SwapTransaction,
    ) -> Dict[str, Any]:
        """Estimate transaction fee before sending.

        Args:
            swap_tx: SwapTransaction to estimate

        Returns:
            Dict with fee estimates in lamports and SOL
        """
        try:
            from solana.rpc.async_api import AsyncClient
            from solders.transaction import VersionedTransaction
            from solders.message import to_bytes_versioned
        except ImportError:
            # Return estimate from swap tx
            return {
                "base_fee_lamports": 5000,
                "priority_fee_lamports": swap_tx.prioritization_fee_lamports,
                "total_lamports": 5000 + swap_tx.prioritization_fee_lamports,
                "total_sol": (5000 + swap_tx.prioritization_fee_lamports) / 1e9,
            }

        try:
            import base58
            tx_bytes = base58.b58decode(swap_tx.swap_transaction)
            vtx = VersionedTransaction.from_bytes(tx_bytes)
            msg_bytes = to_bytes_versioned(vtx.message)
            msg_b64 = base64.b64encode(msg_bytes).decode("utf-8")

            async with AsyncClient(self.config.rpc_url) as rpc:
                # getFeeForMessage returns base fee
                resp = await rpc._provider.make_request(
                    "getFeeForMessage",
                    [msg_b64, {"commitment": "processed"}],
                )
                base_fee = resp.get("result", {}).get("value", 5000)

                total = base_fee + swap_tx.prioritization_fee_lamports

                return {
                    "base_fee_lamports": base_fee,
                    "priority_fee_lamports": swap_tx.prioritization_fee_lamports,
                    "total_lamports": total,
                    "total_sol": total / 1e9,
                }

        except Exception as e:
            logger.warning(f"Fee estimation failed: {e}")
            return {
                "base_fee_lamports": 5000,
                "priority_fee_lamports": swap_tx.prioritization_fee_lamports,
                "total_lamports": 5000 + swap_tx.prioritization_fee_lamports,
                "total_sol": (5000 + swap_tx.prioritization_fee_lamports) / 1e9,
                "error": str(e),
            }


async def execute_swap(
    config: BagsConfig,
    input_mint: str,
    output_mint: str,
    amount: int,
    slippage_bps: int = 100,
    dry_run: bool = True,
) -> SwapResult:
    """Execute a complete swap from quote to confirmation.

    Args:
        config: Bags API configuration
        input_mint: Input token mint
        output_mint: Output token mint
        amount: Amount in smallest units
        slippage_bps: Slippage tolerance in basis points
        dry_run: If True, only get quote without executing

    Returns:
        SwapResult with execution details
    """
    client = BagsAPIClient(config)
    executor = SolanaTransactionExecutor(config)

    # Get quote
    quote = await client.get_quote(
        input_mint=input_mint,
        output_mint=output_mint,
        amount=amount,
        slippage_bps=slippage_bps,
    )

    logger.info(
        f"Quote: {amount} -> {quote.out_amount} "
        f"(min: {quote.min_out_amount}, impact: {quote.price_impact_pct:.4f}%)"
    )

    if dry_run:
        return SwapResult(
            success=True,
            input_amount=amount,
            output_amount=quote.out_amount,
        )

    # Build transaction
    swap_tx = await client.build_swap_transaction(
        quote=quote,
        user_public_key=executor.public_key,
    )

    logger.info(
        f"Swap tx built: compute={swap_tx.compute_unit_limit}, "
        f"priority_fee={swap_tx.prioritization_fee_lamports} lamports"
    )

    # Execute
    result = await executor.sign_and_send(swap_tx)

    if result.success:
        logger.info(f"Swap successful: {result.signature}")
    else:
        logger.error(f"Swap failed: {result.error}")

    return result
