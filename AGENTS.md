use uv pip NEVER just pip

fully finish tasks eg if it means install uv pip packages, write the tests and run them then run the related benchmarks for real with long timeouts - dont give up

code is requiring a lot of thought here as its a production trading bot

try do as much work as you can so dont just give up on installing packages - add them to pyproject.toml uv sync and install -e toto/ too just do things and get stuff tested then simulated properly all the way done

write tests/test a lot while developing - use tools 100s of tool calls is great

Ensure every code modification strictly preserves correctness, minimality of change, and robustly handles edge/corner cases related to the problem statement. ok use simple code structures like functions not complex inheritence.

Avoid blanket or “quick fix” solutions that might hide errors or unintentionally discard critical information; always strive to diagnose and address root-causes, not merely symptoms or side-effects.

Where input normalization is necessary - for types, iterables, containers, or input shapes - do so only in a way that preserves API contracts, allows for extensibility, and maintains invariance across all supported data types, including Python built-ins and major library types. can put any re usable utils in src/ and test them

All error/warning messages, exceptions, and documentation updates must be technically accurate, actionable, match the conventions of the host codebase, and be kept fully in sync with new or changed behavior.

Backwards and forwards compatibility: Changes must account for code used in diverse environments (e.g., different Python versions, framework/ORM versions, or platforms), and leverage feature detection where possible to avoid breaking downstream or legacy code.

Refactorings and bugfixes must never silently discard, mask, or change user data, hooks, plugin registrations, or extension points; if a migration or transformation is required, ensure it is invertible/idempotent where possible

use latest tactics in terms of machine learning can see nanochat/ for some good practice

instead of reconfirming with me just do it - you are probably right and yea i can always roll back thats fine lets just do it.
