"""Robust Solana RPC client with Helius support, rate limiting, and fallback.

Features:
- Multiple RPC endpoints with automatic fallback
- Exponential backoff for rate limits (429 errors)
- Transaction confirmation via polling (not SDK's confirm_transaction)
- Balance caching to reduce RPC calls
- Priority fee estimation via Helius
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

# Try to get Helius API key
try:
    from env_real import HELIUS_API_KEY
except ImportError:
    HELIUS_API_KEY = os.getenv("HELIUS_API_KEY", "")


@dataclass
class RPCEndpoint:
    """Configuration for an RPC endpoint."""

    name: str
    url: str
    priority: int = 0  # Lower = higher priority
    rate_limit_rps: float = 10.0  # Requests per second limit
    is_helius: bool = False

    # Runtime state
    last_request_time: float = 0.0
    consecutive_failures: int = 0
    backoff_until: float = 0.0


@dataclass
class CachedBalance:
    """Cached balance with expiry."""

    balance_lamports: int
    cached_at: datetime
    ttl_seconds: float = 10.0

    @property
    def is_expired(self) -> bool:
        return datetime.utcnow() > self.cached_at + timedelta(seconds=self.ttl_seconds)


@dataclass
class RPCConfig:
    """Configuration for RPC client."""

    # Retry settings
    max_retries: int = 5
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 30.0
    backoff_multiplier: float = 2.0

    # Rate limiting
    min_request_interval_seconds: float = 0.1

    # Confirmation settings
    confirmation_timeout_seconds: float = 60.0
    confirmation_poll_interval_seconds: float = 0.5

    # Cache settings
    balance_cache_ttl_seconds: float = 10.0


class RobustRPCClient:
    """Robust Solana RPC client with fallback and rate limiting."""

    def __init__(
        self,
        primary_rpc_url: str = "https://api.mainnet-beta.solana.com",
        helius_api_key: str = "",
        config: Optional[RPCConfig] = None,
    ):
        self.config = config or RPCConfig()
        self._endpoints: List[RPCEndpoint] = []
        self._balance_cache: Dict[str, CachedBalance] = {}
        self._request_id = 0

        # Add Helius endpoint if API key provided (highest priority)
        helius_key = helius_api_key or HELIUS_API_KEY
        if helius_key:
            self._endpoints.append(RPCEndpoint(
                name="helius",
                url=f"https://mainnet.helius-rpc.com/?api-key={helius_key}",
                priority=0,
                rate_limit_rps=50.0,  # Helius has higher limits
                is_helius=True,
            ))
            logger.info("Helius RPC endpoint configured (primary)")

        # Add primary/fallback endpoint
        self._endpoints.append(RPCEndpoint(
            name="primary",
            url=primary_rpc_url,
            priority=1 if helius_key else 0,
            rate_limit_rps=10.0,
        ))

        # Sort by priority
        self._endpoints.sort(key=lambda e: e.priority)

        logger.info(f"RPC client initialized with {len(self._endpoints)} endpoints")

    def _get_available_endpoints(self) -> List[RPCEndpoint]:
        """Get endpoints that aren't in backoff."""
        now = time.time()
        available = [e for e in self._endpoints if e.backoff_until <= now]

        if not available:
            # All in backoff - use the one with shortest remaining backoff
            return [min(self._endpoints, key=lambda e: e.backoff_until)]

        return available

    async def _rate_limit(self, endpoint: RPCEndpoint) -> None:
        """Apply rate limiting for an endpoint."""
        now = time.time()
        min_interval = 1.0 / endpoint.rate_limit_rps

        time_since_last = now - endpoint.last_request_time
        if time_since_last < min_interval:
            await asyncio.sleep(min_interval - time_since_last)

        endpoint.last_request_time = time.time()

    def _apply_backoff(self, endpoint: RPCEndpoint) -> None:
        """Apply exponential backoff to an endpoint."""
        endpoint.consecutive_failures += 1
        delay = min(
            self.config.base_delay_seconds * (self.config.backoff_multiplier ** endpoint.consecutive_failures),
            self.config.max_delay_seconds,
        )
        endpoint.backoff_until = time.time() + delay
        logger.warning(f"RPC {endpoint.name}: backing off for {delay:.1f}s (failures: {endpoint.consecutive_failures})")

    def _reset_backoff(self, endpoint: RPCEndpoint) -> None:
        """Reset backoff state on success."""
        endpoint.consecutive_failures = 0
        endpoint.backoff_until = 0.0

    async def call(
        self,
        method: str,
        params: Optional[List[Any]] = None,
        timeout: float = 30.0,
    ) -> Any:
        """Make an RPC call with automatic retry and fallback.

        Args:
            method: RPC method name
            params: RPC parameters
            timeout: Request timeout in seconds

        Returns:
            RPC result

        Raises:
            RuntimeError: If all retries exhausted
        """
        self._request_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params or [],
        }

        last_error = None

        for attempt in range(self.config.max_retries):
            endpoints = self._get_available_endpoints()

            for endpoint in endpoints:
                try:
                    await self._rate_limit(endpoint)

                    async with httpx.AsyncClient(timeout=timeout) as client:
                        response = await client.post(endpoint.url, json=payload)

                        # Handle rate limiting
                        if response.status_code == 429:
                            logger.warning(f"RPC {endpoint.name}: rate limited (429)")
                            self._apply_backoff(endpoint)
                            continue

                        response.raise_for_status()
                        data = response.json()

                        if "error" in data:
                            error = data["error"]
                            error_msg = error.get("message", str(error))

                            # Check for rate limit in error message
                            if "rate" in error_msg.lower() or "limit" in error_msg.lower():
                                logger.warning(f"RPC {endpoint.name}: rate limit in response")
                                self._apply_backoff(endpoint)
                                continue

                            raise RuntimeError(f"RPC error: {error_msg}")

                        self._reset_backoff(endpoint)
                        return data.get("result")

                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429:
                        self._apply_backoff(endpoint)
                        last_error = e
                        continue
                    raise

                except Exception as e:
                    last_error = e
                    logger.warning(f"RPC {endpoint.name} failed: {e}")
                    self._apply_backoff(endpoint)

            # Wait before retry
            if attempt < self.config.max_retries - 1:
                delay = self.config.base_delay_seconds * (self.config.backoff_multiplier ** attempt)
                logger.debug(f"Retrying in {delay:.1f}s (attempt {attempt + 2}/{self.config.max_retries})")
                await asyncio.sleep(delay)

        raise RuntimeError(f"RPC call failed after {self.config.max_retries} attempts: {last_error}")

    async def get_balance(self, pubkey: str, use_cache: bool = True) -> int:
        """Get SOL balance in lamports with caching.

        Args:
            pubkey: Base58 public key
            use_cache: Whether to use cached value if available

        Returns:
            Balance in lamports
        """
        # Check cache
        if use_cache and pubkey in self._balance_cache:
            cached = self._balance_cache[pubkey]
            if not cached.is_expired:
                return cached.balance_lamports

        result = await self.call("getBalance", [pubkey, {"commitment": "confirmed"}])
        balance = result.get("value", 0)

        # Cache it
        self._balance_cache[pubkey] = CachedBalance(
            balance_lamports=balance,
            cached_at=datetime.utcnow(),
            ttl_seconds=self.config.balance_cache_ttl_seconds,
        )

        return balance

    async def get_token_accounts_by_owner(
        self,
        owner: str,
        mint: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get SPL token accounts for an owner.

        Args:
            owner: Owner public key
            mint: Optional mint filter

        Returns:
            List of token account info
        """
        if mint:
            filter_param = {"mint": mint}
        else:
            filter_param = {"programId": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"}

        result = await self.call(
            "getTokenAccountsByOwner",
            [owner, filter_param, {"encoding": "jsonParsed"}],
        )

        return result.get("value", [])

    async def send_raw_transaction(
        self,
        tx_bytes: bytes,
        skip_preflight: bool = False,
    ) -> str:
        """Send a raw transaction.

        Args:
            tx_bytes: Serialized transaction bytes
            skip_preflight: Skip preflight simulation

        Returns:
            Transaction signature
        """
        import base64
        tx_b64 = base64.b64encode(tx_bytes).decode("utf-8")

        result = await self.call(
            "sendTransaction",
            [tx_b64, {
                "encoding": "base64",
                "skipPreflight": skip_preflight,
                "preflightCommitment": "confirmed",
            }],
        )

        return result

    async def get_signature_statuses(
        self,
        signatures: List[str],
        search_history: bool = True,
    ) -> List[Optional[Dict[str, Any]]]:
        """Get signature statuses.

        Args:
            signatures: List of transaction signatures
            search_history: Search transaction history

        Returns:
            List of status dicts (or None for not found)
        """
        result = await self.call(
            "getSignatureStatuses",
            [signatures, {"searchTransactionHistory": search_history}],
        )

        return result.get("value", [])

    async def wait_for_confirmation(
        self,
        signature: str,
        commitment: str = "confirmed",
        timeout_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Wait for transaction confirmation via polling.

        This is more reliable than the SDK's confirm_transaction method
        which can hit rate limits during polling.

        Args:
            signature: Transaction signature
            commitment: Desired commitment level (confirmed/finalized)
            timeout_seconds: Timeout (default from config)

        Returns:
            Confirmation status dict

        Raises:
            TimeoutError: If confirmation times out
            RuntimeError: If transaction failed
        """
        timeout = timeout_seconds or self.config.confirmation_timeout_seconds
        deadline = time.time() + timeout
        poll_interval = self.config.confirmation_poll_interval_seconds

        logger.debug(f"Waiting for confirmation: {signature[:20]}... (timeout={timeout}s)")

        while time.time() < deadline:
            try:
                statuses = await self.get_signature_statuses([signature])
                status = statuses[0] if statuses else None

                if status:
                    # Check for error
                    if status.get("err") is not None:
                        raise RuntimeError(f"Transaction failed: {status['err']}")

                    conf_status = status.get("confirmationStatus", "")

                    # Check commitment reached
                    if commitment == "confirmed" and conf_status in ("confirmed", "finalized"):
                        logger.debug(f"Transaction confirmed: {conf_status}")
                        return status

                    if commitment == "finalized" and conf_status == "finalized":
                        logger.debug("Transaction finalized")
                        return status

                    logger.debug(f"Confirmation status: {conf_status}, waiting for {commitment}")

            except RuntimeError:
                raise
            except Exception as e:
                logger.warning(f"Confirmation poll error (will retry): {e}")

            await asyncio.sleep(poll_interval)
            # Increase poll interval to reduce rate limit risk
            poll_interval = min(poll_interval * 1.2, 2.0)

        raise TimeoutError(f"Confirmation timed out after {timeout}s: {signature}")

    async def get_transaction(
        self,
        signature: str,
        max_supported_version: int = 0,
    ) -> Optional[Dict[str, Any]]:
        """Get transaction details.

        Args:
            signature: Transaction signature
            max_supported_version: Max transaction version

        Returns:
            Transaction details or None if not found
        """
        result = await self.call(
            "getTransaction",
            [signature, {
                "encoding": "jsonParsed",
                "maxSupportedTransactionVersion": max_supported_version,
            }],
        )

        return result

    async def get_priority_fee_estimate(
        self,
        account_keys: List[str],
    ) -> Dict[str, int]:
        """Get priority fee estimate (Helius-specific).

        Args:
            account_keys: Account keys involved in transaction

        Returns:
            Dict with fee levels in microlamports
        """
        # Only works with Helius
        helius_endpoint = next((e for e in self._endpoints if e.is_helius), None)

        if not helius_endpoint:
            # Return default estimates
            return {
                "min": 1000,
                "low": 10000,
                "medium": 100000,
                "high": 500000,
                "veryHigh": 1000000,
            }

        try:
            # Use Helius endpoint directly for this call
            self._request_id += 1
            payload = {
                "jsonrpc": "2.0",
                "id": self._request_id,
                "method": "getPriorityFeeEstimate",
                "params": [{
                    "accountKeys": account_keys,
                    "options": {"includeAllPriorityFeeLevels": True},
                }],
            }

            await self._rate_limit(helius_endpoint)

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(helius_endpoint.url, json=payload)
                data = response.json()

                if "result" in data:
                    return data["result"].get("priorityFeeLevels", {})

        except Exception as e:
            logger.warning(f"Priority fee estimate failed: {e}")

        return {
            "min": 1000,
            "low": 10000,
            "medium": 100000,
            "high": 500000,
            "veryHigh": 1000000,
        }

    def invalidate_balance_cache(self, pubkey: Optional[str] = None) -> None:
        """Invalidate balance cache.

        Args:
            pubkey: Specific pubkey to invalidate, or None for all
        """
        if pubkey:
            self._balance_cache.pop(pubkey, None)
        else:
            self._balance_cache.clear()


# Singleton for shared RPC client
_shared_rpc_client: Optional[RobustRPCClient] = None


def get_shared_rpc_client(
    primary_rpc_url: str = "https://api.mainnet-beta.solana.com",
    helius_api_key: str = "",
) -> RobustRPCClient:
    """Get or create a shared RPC client instance.

    Args:
        primary_rpc_url: Primary RPC URL
        helius_api_key: Helius API key

    Returns:
        Shared RPC client instance
    """
    global _shared_rpc_client

    if _shared_rpc_client is None:
        _shared_rpc_client = RobustRPCClient(
            primary_rpc_url=primary_rpc_url,
            helius_api_key=helius_api_key,
        )

    return _shared_rpc_client
