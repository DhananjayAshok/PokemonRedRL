from utils.parameter_handling import load_parameters, compute_secondary_parameters
from utils import log_error, log_info, log_warn
import click
from huggingface_hub import HfApi
import os

loaded_parameters = load_parameters()



@click.command()
@click.option("--private", is_flag=True, default=False, help="Whether the repo should be private.")
@click.pass_obj
def create_hub_repo(parameters, private):
    """
    Create a new repo on the Hugging Face Hub.
    """
    repo_namespace = parameters["huggingface_repo_namespace"]
    repo_name = parameters["huggingface_repo_name"]
    repo_id = f"{repo_namespace}/{repo_name}"
    api = parameters["api"]
    api.create_repo(
    repo_id=repo_id,
    repo_type="dataset",
    exist_ok=True, # Won't raise an error if the repo already exists
    private=private)
    log_info(f"Successfully created repo {repo_id} on the Hugging Face Hub.", parameters)


@click.command()
@click.pass_obj
def setup_sync(parameters):
    """
    Set up the local sync directory to sync with the specified repo on the Hugging Face Hub.
    """
    repo_namespace = parameters["huggingface_repo_namespace"]
    repo_name = parameters["huggingface_repo_name"]    
    repo_id = f"{repo_namespace}/{repo_name}"
    sync_dir = os.path.abspath(parameters["sync_dir"])
    if not os.path.exists(sync_dir):
        log_error(f"Sync directory {sync_dir} does not exist, please check your configuration.", parameters)
    api = parameters["api"]
    api.snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=sync_dir)
    log_info(f"Tried to sync repo at {repo_namespace}/{repo_name} with directory {sync_dir}. Check output above for success", parameters)


@click.command()
@click.pass_obj
def push_data_to_hub(parameters):
    """
    In general how this works is:
        1. Check if repo exists. If it does, first pull and write it to sync directory (i.e. root_dir)
    """
    repo_namespace = parameters["huggingface_repo_namespace"]
    repo_name = parameters["huggingface_repo_name"]
    api = parameters["api"]
    sync_dir = os.path.abspath(parameters["sync_dir"])
    repo_id = f"{repo_namespace}/{repo_name}"
    api.upload_folder(repo_id=repo_id, repo_type="dataset", folder_path=sync_dir)
    log_info(f"Successfully pushed data from {sync_dir} to repo {repo_namespace}/{repo_name} on the Hugging Face Hub.", parameters)


@click.group()
@click.option("--huggingface_repo_namespace", type=str, default=None, help="The namespace (user or org) on the hub where the repo is located.")
@click.option("--huggingface_repo_name", type=str, default=None, help="The name of the repo on the hub.")
@click.pass_context
def main(ctx, **input_parameters):
    if "huggingface_repo_namespace" not in loaded_parameters:
        if input_parameters["huggingface_repo_namespace"] is not None:
            loaded_parameters["huggingface_repo_namespace"] = input_parameters["huggingface_repo_namespace"]
        else:
            log_error("huggingface_repo_namespace must be specified either in the config file or as a command line argument.", loaded_parameters)
    if "huggingface_repo_name" not in loaded_parameters:
        if input_parameters["huggingface_repo_name"] is not None:
            loaded_parameters["huggingface_repo_name"] = input_parameters["huggingface_repo_name"]
        else:
            log_error("huggingface_repo_name must be specified either in the config file or as a command line argument.", loaded_parameters)
    compute_secondary_parameters(loaded_parameters)
    api = HfApi()
    loaded_parameters["api"] = api
    ctx.obj = loaded_parameters


main.add_command(create_hub_repo, name="init")
main.add_command(setup_sync, name="pull")
main.add_command(push_data_to_hub, name="push")

if __name__ == "__main__":
    main()
