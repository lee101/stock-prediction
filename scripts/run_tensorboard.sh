#!/usr/bin/env bash
set -euo pipefail

# Allow overriding the target with TENSORBOARD_MAX_OPEN_FILES; default to 65536.
TARGET_LIMIT="${TENSORBOARD_MAX_OPEN_FILES:-65536}"

if [[ -z "${TARGET_LIMIT}" ]]; then
  echo "TENSORBOARD_MAX_OPEN_FILES must not be empty if set." >&2
  exit 1
fi

if ! [[ "${TARGET_LIMIT}" =~ ^[0-9]+$ ]]; then
  echo "TENSORBOARD_MAX_OPEN_FILES must be an integer (received: ${TARGET_LIMIT})." >&2
  exit 1
fi

HARD_LIMIT="$(ulimit -Hn)"

if [[ "${HARD_LIMIT}" != "unlimited" ]]; then
  if (( TARGET_LIMIT > HARD_LIMIT )); then
    echo "Warning: requested ${TARGET_LIMIT} descriptors but the hard limit is ${HARD_LIMIT}; using the hard limit instead." >&2
    TARGET_LIMIT="${HARD_LIMIT}"
  fi
fi

CURRENT_LIMIT="$(ulimit -n)"

if [[ "${CURRENT_LIMIT}" != "unlimited" ]]; then
  # Only raise the file descriptor ceiling when the current limit is below the target.
  if (( CURRENT_LIMIT < TARGET_LIMIT )); then
    if ! RAISE_ERR="$(ulimit -n "${TARGET_LIMIT}" 2>&1)"; then
      echo "Warning: unable to raise open files limit to ${TARGET_LIMIT}: ${RAISE_ERR}" >&2
    fi
  fi
fi

exec tensorboard "$@"
