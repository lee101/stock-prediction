#!/usr/bin/env python3
import os
from datetime import datetime, timezone
import sys

try:
    from alpaca.trading.client import TradingClient
    from alpaca.common.exceptions import APIError
except ImportError:
    print("Failed to import Alpaca SDK components. Please ensure 'alpaca-trade-api' is installed.")
    sys.exit(1)

# Attempt to import API keys from env_real.py, fallback to environment variables
try:
    from env_real import ALP_KEY_ID, ALP_SECRET_KEY, ALP_ENDPOINT
    print("Loaded credentials from env_real.py")
except ImportError:
    print("Could not import from env_real.py. Trying environment variables.")
    ALP_KEY_ID = os.getenv("ALP_KEY_ID")
    ALP_SECRET_KEY = os.getenv("ALP_SECRET_KEY")
    # Default to paper trading endpoint if not specified
    ALP_ENDPOINT = os.getenv("ALP_ENDPOINT", "https://paper-api.alpaca.markets") 

if not ALP_KEY_ID or not ALP_SECRET_KEY:
    print("Error: API keys (ALP_KEY_ID, ALP_SECRET_KEY) not found.")
    print("Please ensure they are set in env_real.py or as environment variables.")
    sys.exit(1)

# Determine if paper trading based on the endpoint
# Production URL: "https://api.alpaca.markets"
# Paper URL: "https://paper-api.alpaca.markets"
is_paper_trading = ALP_ENDPOINT != "https://api.alpaca.markets"
trading_env = "paper" if is_paper_trading else "live"
print(f"Configuring Alpaca client for {trading_env} trading environment.")

alpaca_api = TradingClient(
    ALP_KEY_ID,
    ALP_SECRET_KEY,
    paper=is_paper_trading
)

def get_current_ip_address():
    """
    Prompts the user for their current IP address.
    In a more automated scenario, a service like httpbin.org/ip could be used,
    but that requires an external HTTP request and library.
    """
    while True:
        ip_address = input("Please enter your current public IP address (e.g., 185.13.21.99): ").strip()
        # Basic validation for IP format (not exhaustive)
        parts = ip_address.split('.')
        if len(parts) == 4 and all(part.isdigit() and 0 <= int(part) <= 255 for part in parts):
            return ip_address
        else:
            print("Invalid IP address format. Please try again.")

def enable_crypto_trading_for_account():
    """
    Checks crypto status and attempts to enable it by signing the agreement.
    """
    try:
        print("Fetching account details...")
        account = alpaca_api.get_account()
        
        # Use direct attribute access for TradeAccount model
        print(f"  Account ID: {account.id}")
        print(f"  Account Number: {account.account_number}")
        print(f"  Status: {account.status}")
        print(f"  Current Crypto Status: {account.crypto_status}")

        if account.crypto_status == "ACTIVE":
            print("Crypto trading is already ACTIVE for this account.")
            return True
        elif account.crypto_status == "SUBMITTED":
            print("Crypto agreement has already been SUBMITTED. Waiting for Alpaca to activate.")
            return True
        # Other statuses could be None, INACTIVE, REJECTED_CLEARING, REJECTED_COMPLIANCE etc.
        # We proceed if not ACTIVE or SUBMITTED.

        print("\nAttempting to enable crypto trading...")
        
        ip_address = get_current_ip_address()
        signed_at_dt = datetime.now(timezone.utc)
        # Format: 2023-01-01T18:13:44Z (with Z for UTC)
        signed_at_iso = signed_at_dt.strftime('%Y-%m-%dT%H:%M:%SZ')

        agreement_payload = {
            "agreements": [
                {
                    "agreement": "crypto_agreement",
                    "signed_at": signed_at_iso,
                    "ip_address": ip_address
                }
            ]
        }
        
        print(f"  Prepared payload for agreement: {agreement_payload}")
        print(f"  Targeting endpoint: PATCH /v1/accounts/{account.id}")

        # Using the internal _request method to specify API version V1
        # The path should be relative to the base URL, and the version prefix will be handled.
        # e.g. path="/v1/accounts/{account_id}"
        response_data = alpaca_api._request(
            method="PATCH",
            path=f"/v1/accounts/{account.id}", 
            data=agreement_payload, # The _request method handles json for data in PATCH
        )
        
        print("\nSuccessfully submitted crypto agreement.")
        print("  Response from PATCH request:")
        # Assuming response_data is a dictionary (RawData behaves like one)
        if isinstance(response_data, dict):
            for key, value in response_data.items():
                print(f"    {key}: {value}")
            
            current_crypto_status = response_data.get("crypto_status")
            if current_crypto_status == "SUBMITTED" or current_crypto_status == "ACTIVE":
                print(f"\nCrypto agreement successfully submitted. Status is now: {current_crypto_status}")
                print("Please allow some time for Alpaca to process and activate crypto trading if status is SUBMITTED.")
            elif "agreements" in response_data and isinstance(response_data.get("agreements"), list):
                 updated_agreement = next((item for item in response_data["agreements"] if isinstance(item, dict) and item.get("agreement") == "crypto_agreement"), None)
                 if updated_agreement:
                     print(f"\nCrypto agreement details updated. Signed at: {updated_agreement.get('signed_at')}")
                     print("Check your Alpaca dashboard or re-run this script after some time to confirm crypto_status is ACTIVE.")
                 else:
                    print("\nAgreement submitted, but couldn't confirm crypto_agreement details in response. Please check dashboard.")
            else:
                print("\nCrypto agreement submitted, but the response format was unexpected or crypto_status not found. Please check your Alpaca dashboard.")
        else:
            print(f"\nResponse data was not a dictionary as expected: {response_data}")
            print("Please check your Alpaca dashboard.")
        return True

    except APIError as e:
        print(f"\nAlpaca API Error: {e}") # str(e) usually gives a good summary
        print("  Failed to enable crypto trading.")
        if hasattr(e, 'status_code'): # Not all APIError instances might have these
            print(f"  Status Code: {e.status_code}")
        if hasattr(e, 'code'):
            print(f"  Error Code: {e.code}")
        # For the raw message, str(e) is often better or e.args[0]
        # print(f"  Message: {e.message}") # This was a linter error
        if hasattr(e, 'response') and e.response is not None and hasattr(e.response, 'text'):
            print(f"  Raw response: {e.response.text}")
        return False
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Crypto Agreement Signer for Alpaca")
    print("==================================")
    print("This script will attempt to sign the crypto agreement for your Alpaca account.")
    print("Disclaimer: Ensure you understand the terms of the crypto agreement.")
    
    confirmation = input("Do you want to proceed? (yes/no): ").strip().lower()
    if confirmation == 'yes':
        enable_crypto_trading_for_account()
    else:
        print("Operation cancelled by the user.")
    
    print("\nScript finished.") 