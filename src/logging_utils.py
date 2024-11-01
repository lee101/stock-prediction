import sys
import logging
from datetime import datetime
import pytz
from logging.handlers import RotatingFileHandler

class EDTFormatter(logging.Formatter):
    """Formatter that includes both UTC and Eastern time with colored output."""
    def __init__(self):
        super().__init__()
        try:
            self.local_tz = pytz.timezone('US/Eastern')
        except pytz.exceptions.UnknownTimeZoneError:
            print("Warning: US/Eastern timezone not found, falling back to UTC")
            self.local_tz = pytz.UTC

        self.level_colors = {
            "DEBUG": "\033[36m",
            "INFO": "\033[32m",
            "WARNING": "\033[33m",
            "ERROR": "\033[31m",
            "CRITICAL": "\033[35m"
        }
        self.reset_color = "\033[0m"

    def format(self, record):
        try:
            # Get UTC time
            utc_time = datetime.fromtimestamp(record.created, pytz.UTC).strftime('%Y-%m-%d %H:%M:%S %Z')
            # Get local time
            local_time = datetime.now(self.local_tz).strftime('%Y-%m-%d %H:%M:%S %Z')
            
            level_color = self.level_colors.get(record.levelname, "")
            
            # Handle dict-like objects that may not support direct string formatting
            message = str(record.msg)
            if isinstance(record.msg, dict):
                message = str(record.msg)
            elif hasattr(record.msg, '__dict__'):
                message = str(record.msg.__dict__)
            
            return f"{utc_time} | {local_time} | {level_color}{record.levelname}{self.reset_color} | {message}"
        except Exception as e:
            # Fallback formatting if something goes wrong
            return f"[ERROR FORMATTING LOG] {str(record.msg)}"

def setup_logging(log_file: str) -> logging.Logger:
    """Configure logging to output to both stdout and a file with EDT formatting."""
    try:
        # Create logger
        logger = logging.getLogger('main_logger')
        logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers
        logger.handlers.clear()
        
        # Create formatters
        formatter = EDTFormatter()
        
        # Create and configure stdout handler
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.INFO)
        stdout_handler.setFormatter(formatter)
        
        # Create and configure file handler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=500 * 1024 * 1024,  # 500MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(stdout_handler)
        logger.addHandler(file_handler)
        
        return logger
    except Exception as e:
        print(f"Error setting up logging: {str(e)}")
        raise