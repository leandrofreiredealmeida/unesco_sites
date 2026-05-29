import sys
import logging
from pathlib import Path

logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

log_dir = Path(__file__).parents[2] / "logs"
log_dir.mkdir(exist_ok=True)
log_filepath = log_dir / "logging.log"

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger("unescositeslogger")