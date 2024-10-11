import os
import sys
import logging
from datetime import datetime

# Generate the log file name based on the current date and time
LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

#logging_str = "[ %(asctime)s ] %(lineno)d: %(module)s- %(levelname)s -  %(message)s"
# Define the detailed logging format string
logging_str = (
    "[%(asctime)s] "       # Timestamp when the log message is created
    "[%(levelname)s] "     # Logging level (e.g., INFO, DEBUG, ERROR)
    "[File: %(filename)s] " # File name where the log call was made
    "[Line: %(lineno)d] "  # Line number in the source code where the log call was made
    #"[Module: %(module)s] "  # Module (filename) where the log call was made
    #"[Function: %(funcName)s] "  # Function name where the log call was made
    "- %(message)s"        # The actual log message
)

log_dir = "logs"

log_filepath = os.path.join(log_dir,LOG_FILE)

# Creating the log directory if it doesn't exist
os.makedirs(log_dir, exist_ok=True)


# Configureing the logging module with basic settings
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO/WARNING/ERROR
    format=logging_str,  # Use the custom logging format defined above

    handlers=[
        logging.FileHandler(log_filepath),  # Write log messages to the file
        logging.StreamHandler(sys.stdout)   # Also output log messages to the console
    ]
)
# Create a custom logger named "mutimodalLogger"
logger = logging.getLogger("mutimodallogger")
