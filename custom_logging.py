import logging, json
from datetime import datetime


class CustomJsonFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S'),
            'level': record.levelname,
            'message': record.getMessage()
        }
        return json.dumps(log_entry)


def setup_logger(name="my-logger", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:  # prevents duplicate handlers on repeated calls/imports
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(CustomJsonFormatter())
        logger.addHandler(streamHandler)

        fileHandler = logging.FileHandler('webservice.log', mode='a')
        fileHandler.setFormatter(CustomJsonFormatter())
        logger.addHandler(fileHandler)

    return logger


# ==============================================================================
# HOW PYTHON LOG LEVELS FILTER MESSAGES
# ==============================================================================
# Rule: Setting a level acts as a MINIMUM threshold.
#       It captures that specific level and anything HIGHER (more severe).
#       It completely BLOCKS anything LOWER (less severe).
#
# Severity Scale & Behavior:
# -------------------------
# [50] CRITICAL --> Captured if level is CRITICAL, ERROR, WARNING, INFO, or DEBUG
# [40] ERROR    --> Captured if level is ERROR, WARNING, INFO, or DEBUG
# [30] WARNING  --> Captured if level is WARNING, INFO, or DEBUG (Default Level)
# ------------------------------------------------------------------------------
# [20] INFO     --> Captured ONLY if level is INFO or DEBUG
# [10] DEBUG    --> Captured ONLY if level is DEBUG
# ==============================================================================
# EXAMPLES:
# ------------------------------------------------------------------------------
# 1. If you set level to WARNING:
#    -> Captured: WARNING, ERROR, CRITICAL
#    -> Hidden:   INFO, DEBUG (Too low severity!)
#
# 2. If you set level to INFO:
#    -> Captured: INFO, WARNING, ERROR, CRITICAL
#    -> Hidden:   DEBUG (Too low severity!)
#
# 3. If you want to see absolutely EVERYTHING, set the level to DEBUG.
# ==============================================================================