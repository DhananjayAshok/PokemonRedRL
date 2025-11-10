import os
import yaml
from utils.fundamental import get_logger


def load_yaml(yaml_path):
    with open(yaml_path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def compute_secondary_parameters(params):
    params["data_dir"] = os.path.join(params["storage_dir"], "data")
    params["model_dir"] = os.path.join(params["storage_dir"], "models")
    params["tmp_dir"] = os.path.join(params["storage_dir"], "tmp")
    params["sync_dir"] = os.path.join(params["storage_dir"], "sync")
    params["log_dir"] = os.path.join(params["results_dir"], "logs")
    params["figure_dir"] = os.path.join(params["results_dir"], "figures")    
    for dirname in ["data_dir", "model_dir", "log_dir", "figure_dir", "tmp_dir", "sync_dir"]:
        if not os.path.exists(params[dirname]):
            os.makedirs(params[dirname])
    if "log_file" not in params:
        log_file = os.path.join(params["log_dir"], "log.txt")
        params["log_file"] = log_file
    else:
        # check if log_file is a child of log_dir, but handle silly // vs / cases
        log_dir_str = params["log_dir"].replace("//", "/")
        log_file_str = params["log_file"].replace("//", "/")
        if not log_file_str.startswith(log_dir_str):
            log_file = os.path.join(params["log_dir"], params["log_file"])
            params["log_file"] = log_file
    logger = get_logger(filename=params["log_file"])
    params["logger"] = logger


def load_parameters(parameters=None):
    """
    Loads the parameters for the project from configs/private_vars.yaml and any other yaml files in the configs directory.

    That is, unless a non None parameters dictionary is passed through, in which case we assume all is good and just return it.

        :return: A dictionary of parameters
    """
    if parameters is not None:
        if "logger" not in parameters: # this is a flag that secondary parameters need to be computed
            compute_secondary_parameters(parameters)
        return parameters
    essential_keys = ["storage_dir", "results_dir", "figure_force_save"]
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    params = {"project_root": project_root}
    logger = get_logger()
    config_files = os.listdir(os.path.join(project_root, "configs"))
    def error(msg):
        logger.error(msg)
        raise ValueError(msg)

    if "private_vars.yaml" not in config_files:
        error("Please create private_vars.yaml in the configs directory")
    for file in config_files:
        if file.endswith(".yaml"):
            configs = load_yaml(os.path.join(project_root, "configs", file))
            for key in configs:
                if key in params:
                    error(f"{key} is present in multiple config files. At least one of which is {file}. Please remove the duplicate")
            params.update(configs)
        else:
            pass

    for key in params:
        if params[key] == "PLACEHOLDER":
            error(f"{key} is currently the placeholder value in private_vars.yaml. Please set it")
    for essential_key in essential_keys:
        if essential_key not in params:
            error(f"Please set {essential_key} in one of the config yamls")
    # check if there are any .py files in storage_dir, if so, log error
    if os.path.exists(params['storage_dir']):
        if any([f.endswith(".py") for f in os.listdir(params["storage_dir"])]):
            logger.warning(f"There are .py files in the storage_dir {params['storage_dir']}. It is recommended to set a path which has nothing else inside it to avoid issues.")
    else:
        os.makedirs(params['storage_dir'])
        logger.info(f"Created storage directory {params['storage_dir']}")
    compute_secondary_parameters(params)
    return params
