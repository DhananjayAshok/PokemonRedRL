# This file contains all the fundamental utilities that do not rely on any other file. 
import os
import logging

def get_logger(level=logging.INFO, filename=None, add_console=True):
    fmt_str = "%(asctime)s, [%(levelname)s, %(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=fmt_str)
    logger = logging.getLogger("PROJECT_NAME")
    if add_console:
        logger.handlers.clear()
        console_handler = logging.StreamHandler()
        log_formatter = logging.Formatter(fmt_str)
        console_handler.setFormatter(log_formatter)
        logger.addHandler(console_handler)
    if filename is not None:
        file_handler = logging.FileHandler(filename, mode="a")
        log_formatter = logging.Formatter(fmt_str)
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)
    if level is not None:
        logger.setLevel(level)
        logger.propagate = False
    return logger

def meta_dict_to_str(meta_dict, print_mode=False, n_indents=1, skip_write_timestamp=True):
    keys = list(meta_dict.keys())
    keys.sort()
    meta_str = ""
    for key in keys:
        if print_mode:
            indent = '\t' * n_indents
            meta_str += f"{indent}{key}: {meta_dict[key]}\n"
        else:
            if skip_write_timestamp and key == "write_timestamp":
                continue
            meta_str += f"{key.lower().strip()}_{str(meta_dict[key]).lower().strip()}"
    return meta_str


def logger_print_dict(logger, meta_dict, n_indents=1):
    meta_dict_str = meta_dict_to_str(meta_dict, print_mode=True, n_indents=n_indents, skip_write_timestamp=False)
    logger.info(meta_dict_str)


def file_makedir(file_path):
    dirname = os.path.dirname(file_path)
    if dirname != "" and not os.path.exists(dirname):
        os.makedirs(dirname)
    return
