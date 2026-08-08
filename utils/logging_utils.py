"""Logging helpers for MoQ-NAS.

``init_log`` extracted verbatim from ``utils/helpers.py`` (Block C of the
refactor roadmap). ``helpers.py`` re-exports it until it becomes a facade
(stage C.6), so both import paths are equivalent. Importing this module
has no side-effects; handlers/files are only created when ``init_log``
is called.
"""
import logging


def init_log(log_level, name, file_path=None):
    """ Initialize a logging.Logger with level *log_level* and name *name*.

    Args:
        log_level: (str) one of 'NONE', 'INFO' or 'DEBUG'.
        name: (str) name of the module initiating the logger (will be the logger name).
        file_path: (str) path to the log file. If None, stdout is used.

    Returns:
        logging.Logger object.
    """

    logger = logging.getLogger(name)
        # Eliminar handlers existentes para evitar duplicación
    if logger.hasHandlers():
        logger.handlers.clear()

    if file_path is None:
        handler = logging.StreamHandler()
    else:
        handler = logging.FileHandler(file_path)

    formatter = logging.Formatter('%(levelname)s: %(module)s: %(asctime)s.%(msecs)03d '
                                '- %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if log_level == 'INFO':
        logger.setLevel(logging.INFO)
    elif log_level == 'DEBUG':
        logger.setLevel(logging.DEBUG)

    return logger
