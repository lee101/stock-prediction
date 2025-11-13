import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


class EDTFormatter(logging.Formatter):
    """Formatter that includes both UTC and Eastern time with colored output."""

    def __init__(self):
        super().__init__()
        self.utc_zone = ZoneInfo("UTC")
        self.local_tz = self._load_zone("US/Eastern", self.utc_zone)
        self.nzdt_zone = self._load_zone("Pacific/Auckland", self.utc_zone)

        self.level_colors = {
            "DEBUG": "\033[36m",
            "INFO": "\033[32m",
            "WARNING": "\033[33m",
            "ERROR": "\033[31m",
            "CRITICAL": "\033[35m"
        }
        self.reset_color = "\033[0m"

    @staticmethod
    def _load_zone(name: str, fallback: ZoneInfo) -> ZoneInfo:
        try:
            return ZoneInfo(name)
        except ZoneInfoNotFoundError:
            print(f"Warning: timezone {name} not found, falling back to {fallback.key if hasattr(fallback, 'key') else 'UTC'}")
            return fallback

    def format(self, record):
        try:
            record_time = datetime.fromtimestamp(record.created, tz=self.utc_zone)
            utc_time = record_time.astimezone(self.utc_zone).strftime('%Y-%m-%d %H:%M:%S %Z')
            local_time = record_time.astimezone(self.local_tz).strftime('%Y-%m-%d %H:%M:%S %Z')
            nzdt_time = record_time.astimezone(self.nzdt_zone).strftime('%Y-%m-%d %H:%M:%S %Z')

            level_color = self.level_colors.get(record.levelname, "")

            # Handle parameter interpolation via logging's standard helper.
            message = record.getMessage()
            if isinstance(record.msg, dict):
                message = str(record.msg)
            elif hasattr(record.msg, "__dict__"):
                message = str(record.msg.__dict__)

            # Get file, function, and line number
            filename = os.path.basename(record.pathname)
            func_name = record.funcName
            line_no = record.lineno

            return f"{utc_time} | {local_time} | {nzdt_time} | {filename}:{func_name}:{line_no} {level_color}{record.levelname}{self.reset_color} | {message}"
        except Exception as e:
            # Fallback formatting if something goes wrong
            return f"[ERROR FORMATTING LOG] {str(record.msg)} - Error: {str(e)}"


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_level(*keys: str, default: str = "INFO") -> int:
    for key in keys:
        value = os.getenv(key)
        if value:
            level = getattr(logging, value.strip().upper(), None)
            if isinstance(level, int):
                return level
    return getattr(logging, default.upper(), logging.INFO)


def get_log_filename(base_name: str, is_hourly: bool = False, is_paper: bool = None) -> str:
    """
    Generate log filename based on trading mode (hourly vs daily) and paper vs live.

    Args:
        base_name: Base log filename (e.g., "trade_stock_e2e.log" or "alpaca_cli.log")
        is_hourly: Whether this is hourly trading (vs daily)
        is_paper: Whether this is paper trading. If None, reads from env PAPER variable

    Returns:
        Modified log filename with appropriate suffixes

    Examples:
        trade_stock_e2e.log (daily live)
        trade_stock_e2e_paper.log (daily paper)
        trade_stock_e2e_hourly.log (hourly live)
        trade_stock_e2e_hourly_paper.log (hourly paper)
    """
    if is_paper is None:
        is_paper = _env_flag("PAPER", default=False)

    # Remove .log extension if present
    if base_name.endswith(".log"):
        base_name = base_name[:-4]

    # Build suffix parts
    suffixes = []
    if is_hourly:
        suffixes.append("hourly")
    if is_paper:
        suffixes.append("paper")

    # Construct final filename
    if suffixes:
        return f"{base_name}_{'_'.join(suffixes)}.log"
    return f"{base_name}.log"


def setup_logging(log_file: str) -> logging.Logger:
    """Configure logging to output to both stdout and a file with optional compact formatting."""
    try:
        # Create logger
        logger_name = os.path.splitext(os.path.basename(log_file))[0]
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)

        # Clear any existing handlers to prevent duplicate logs if called multiple times
        if logger.hasHandlers():
            logger.handlers.clear()

        # Determine formatting strategy
        compact_console = _env_flag("COMPACT_TRADING_LOGS")
        console_formatter = (
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            if compact_console
            else EDTFormatter()
        )
        file_formatter = EDTFormatter()

        console_level = _resolve_level(
            f"{logger_name.upper()}_CONSOLE_LEVEL",
            "TRADING_STDOUT_LEVEL",
            "TRADING_CONSOLE_LEVEL",
            default="INFO",
        )

        # Create and configure stdout handler
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(console_level)
        stdout_handler.setFormatter(console_formatter)

        # Create and configure file handler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=500 * 1024 * 1024,  # 500MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_formatter)

        # Add handlers to logger
        logger.addHandler(stdout_handler)
        logger.addHandler(file_handler)

        # Prevent log messages from propagating to the root logger
        logger.propagate = False

        return logger
    except Exception as e:
        print(f"Error setting up logging for {log_file}: {str(e)}")
        raise
