import logging
import os
from datetime import datetime

# Generate log file name based on current timestamp
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Create logs directory if it doesn't exist
logs_path = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_path, exist_ok=True)

# Full path of the log file
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)


