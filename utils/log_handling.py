from utils.fundamental import meta_dict_to_str
from utils.parameter_handling import load_parameters


def log_error(message, parameters=None):
    parameters = load_parameters(parameters)
    logger = parameters["logger"]
    logger.error(message, stacklevel=2)
    raise ValueError(message)

def log_warn(message, parameters=None):
    parameters = load_parameters(parameters)
    logger = parameters["logger"]
    logger.warn(message, stacklevel=2)

def log_info(message, parameters=None):
    parameters = load_parameters(parameters)
    logger = parameters["logger"]
    logger.info(message, stacklevel=2)

def log_dict(meta_dict, n_indents=1, parameters=None):
    """
    Print a dictionary in a readable format
    """
    parameters = load_parameters(parameters)
    logger = parameters["logger"]
    meta_dict_str = meta_dict_to_str(meta_dict, print_mode=True, n_indents=n_indents, skip_write_timestamp=False)
    logger.info(meta_dict_str, stacklevel=2)
