# Linting and Type Checking Setup

This document describes the linting and type checking setup for the codebase.

## Tools

### Ruff
A fast Python linter and code formatter that combines the functionality of multiple tools (flake8, isort, pyupgrade, etc.).

**Configuration**: `ruff.toml`

### Mypy
A static type checker for Python that helps catch type errors before runtime.

**Configuration**: `pyproject.toml` under `[tool.mypy]`

### Ty
An alternative type checker (already configured in the project).

**Configuration**: `pyproject.toml` under `[tool.ty]`

## Quick Start

### Install Tools
```bash
source .venv/bin/activate
uv pip install ruff mypy
```

### Run Linting
```bash
# Check all files
ruff check .

# Check specific files
ruff check src/env_parsing.py

# Auto-fix issues
ruff check --fix .

# Format code
ruff format .
```

### Run Type Checking
```bash
# Check all files
mypy .

# Check specific files
mypy src/env_parsing.py src/price_calculations.py

# Check with strict mode (for new modules)
mypy --strict src/env_parsing.py
```

## Ruff Configuration

### Enabled Rule Sets
- **F** (Pyflakes) - Undefined names, unused imports
- **E/W** (pycodestyle) - PEP 8 style violations
- **I** (isort) - Import sorting
- **N** (pep8-naming) - Naming conventions
- **UP** (pyupgrade) - Modernize Python code
- **B** (flake8-bugbear) - Find likely bugs
- **A** (flake8-builtins) - Avoid shadowing builtins
- **C4** (flake8-comprehensions) - Improve comprehensions
- **DTZ** (flake8-datetimez) - Timezone-aware datetimes
- **PIE** (flake8-pie) - Misc lints
- **PT** (flake8-pytest-style) - Pytest best practices
- **SIM** (flake8-simplify) - Code simplification
- **PL** (Pylint) - Comprehensive linting
- **RUF** (Ruff-specific) - Ruff's own rules

### Auto-fixable Rules
Ruff can automatically fix issues for these categories:
- Import sorting (I)
- Unused imports (F401)
- Code modernization (UP)
- Comprehension improvements (C4)
- Code simplification (SIM)

### Ignored Rules (Too Noisy)
- **E501** - Line too long (handled by formatter)
- **PLR0913** - Too many arguments
- **PLR0912** - Too many branches
- **PLR0915** - Too many statements
- **PLR2004** - Magic values in comparisons
- **T201** - Print statements (we use both print and logging)
- **ARG001/002** - Unused arguments (often intentional)
- **RET504/505/506** - Unnecessary assignments/else before return

### Per-File Ignores
- `__init__.py` - Allow unused imports (often for re-exports)
- `tests/**` - More lenient (allow unused args, magic values, prints)
- `scripts/**` - Allow print statements

## Mypy Configuration

### General Settings
- Python version: 3.11
- `warn_return_any`: true
- `warn_unused_configs`: true
- `no_implicit_optional`: true
- `warn_redundant_casts`: true
- `warn_unused_ignores`: true
- `strict_equality`: true
- `check_untyped_defs`: true
- `ignore_missing_imports`: true (don't fail on missing stubs)

### Per-Module Overrides

#### Strict Mode (New Modules)
These modules require full type hints:
- `src.env_parsing`
- `src.price_calculations`
- `src.strategy_price_lookup`
- `src.torch_device_utils`

Settings:
- `disallow_untyped_defs`: true
- `disallow_incomplete_defs`: true
- `check_untyped_defs`: true

#### Lenient Mode (Tests)
- `tests.*` - Type hints optional

## Pre-commit Hooks

Pre-commit hooks automatically run linting and formatting before each commit.

### Install Pre-commit
```bash
source .venv/bin/activate
uv pip install pre-commit
pre-commit install
```

### Run Manually
```bash
# Run on all files
pre-commit run --all-files

# Run on specific files
pre-commit run --files src/env_parsing.py

# Skip hooks (not recommended)
git commit --no-verify
```

### Hooks Configured
1. **Ruff linter** - Auto-fix code issues
2. **Ruff formatter** - Auto-format code
3. **Mypy** - Type check src/ directory
4. **Trailing whitespace** - Remove trailing whitespace
5. **End of file fixer** - Ensure files end with newline
6. **YAML/JSON/TOML checker** - Validate config files
7. **Large files checker** - Prevent committing large files (>5MB)
8. **Merge conflict checker** - Detect merge conflict markers
9. **Debug statements** - Catch leftover debugger calls

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Lint and Type Check

on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install uv
          uv pip install ruff mypy
      - name: Run ruff
        run: ruff check .
      - name: Run mypy
        run: mypy src/
```

## Best Practices

### For New Code
1. **Always run ruff and mypy** before committing
2. **Add type hints** to all function signatures
3. **Use modern Python syntax** (e.g., `list[str]` instead of `List[str]`)
4. **Fix issues, don't ignore them** (unless it's a false positive)
5. **Keep functions focused** - if pylint complains about complexity, consider refactoring

### For Existing Code
1. **Low-risk improvements first**:
   - Fix unused imports
   - Sort imports
   - Modernize type hints
   - Add simple type hints
2. **Gradually increase strictness**:
   - Start with `mypy --check-untyped-defs`
   - Move to `--disallow-incomplete-defs`
   - Eventually enable `--strict` for new modules
3. **Use inline ignores sparingly**:
   - Always add a comment explaining why
   - Prefer fixing the issue over ignoring it

## Common Issues

### Ruff: "Undefined name"
**Problem**: Using a variable/function before importing it.
**Solution**: Add the correct import or fix the typo.

### Ruff: "Unused import"
**Problem**: Import is not used in the file.
**Solution**: Remove the import or use it. If it's for re-export, add `# noqa: F401`.

### Mypy: "Function is missing a type annotation"
**Problem**: Function doesn't have type hints.
**Solution**: Add type hints:
```python
def my_func(x: int, y: str) -> bool:
    return len(y) > x
```

### Mypy: "Incompatible return value type"
**Problem**: Function returns a different type than declared.
**Solution**: Fix the return type or the implementation.

### Mypy: "Cannot find implementation or library stub"
**Problem**: Library doesn't have type stubs.
**Solution**: Add to `ignore_missing_imports` or install stubs (`pip install types-*`).

## Migration Strategy

### Phase 1: Low-Risk Improvements (Current)
- ✅ Set up ruff and mypy configurations
- ✅ Enable pre-commit hooks
- ✅ Add type hints to new modules
- ✅ Auto-fix safe issues (imports, formatting, etc.)

### Phase 2: Gradual Strictness Increase
- Add type hints to utility modules in `src/`
- Enable stricter mypy checks per module
- Fix low-hanging type issues
- Document complex types

### Phase 3: Full Type Coverage
- Require type hints in all new code
- Add type hints to core modules
- Enable `--strict` for most modules
- Comprehensive CI checks

## Resources

- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Mypy Documentation](https://mypy.readthedocs.io/)
- [PEP 8 Style Guide](https://peps.python.org/pep-0008/)
- [PEP 484 Type Hints](https://peps.python.org/pep-0484/)
- [Python Type Hints Cheat Sheet](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html)
