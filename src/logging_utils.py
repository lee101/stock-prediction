import sys
from datetime import datetime
import pytz

try:
    from loguru import logger
except ImportError:
    raise ImportError(
        "loguru package is required but not installed. "
        "Please install it using: pip install loguru"
    )

class EDTFormatter:
    """Formatter that includes both UTC and Eastern time with colored output."""
    def __init__(self):
        try:
            self.local_tz = pytz.timezone('US/Eastern')
        except pytz.exceptions.UnknownTimeZoneError:
            print("Warning: US/Eastern timezone not found, falling back to UTC")
            self.local_tz = pytz.UTC

    def __call__(self, record):
        try:
            utc_time = record["time"].strftime('%Y-%m-%d %H:%M:%S %Z')
            local_time = datetime.now(self.local_tz).strftime('%Y-%m-%d %H:%M:%S %Z')
            
            level_colors = {
                "DEBUG": "\033[36m", 
                "INFO": "\033[32m",
                "WARNING": "\033[33m",
                "ERROR": "\033[31m",
                "CRITICAL": "\033[35m"
            }
            reset_color = "\033[0m"
            level_color = level_colors.get(record['level'].name, "")
            
            # Handle dict-like objects that may not support direct string formatting
            message = str(record['message'])
            if isinstance(record['message'], dict):
                message = str(record['message'])
            elif hasattr(record['message'], '__dict__'):
                message = str(record['message'].__dict__)
                
            return f"{utc_time} | {local_time} | {level_color}{record['level'].name}{reset_color} | {message}\n"
        except Exception as e:
            # Fallback formatting if something goes wrong
            return f"[ERROR FORMATTING LOG] {str(record['message'])}\n"

def setup_logging(log_file: str):
    """Configure logging to output to both stdout and a file with EDT formatting."""
    try:
        logger.remove()  # Remove default handler
        
        # Add stdout handler with INFO level
        logger.add(sys.stdout, format=EDTFormatter(), level="INFO", 
                  catch=True)
        
        # Add file handler with DEBUG level to catch everything
        logger.add(log_file, format=EDTFormatter(), level="DEBUG", 
                  backtrace=True, diagnose=True, catch=True,
                  rotation="500 MB")  # Rotate logs when they reach 500MB
        
        return logger
    except Exception as e:
        print(f"Error setting up logging: {str(e)}")
        raise