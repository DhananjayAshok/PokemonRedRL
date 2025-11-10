This project uses [uv](https://docs.astral.sh/uv/guides/projects/) to manage dependencies. One notable difference is that while the project setup with uv is usually done at the root level, we will set up the environment at the [setup](../setup) level instead. 

First, ensure you have installed uv in your python. This project used python 3.XX
```console
pip install --upgrade pip  uv
```


# Replicating Existing UV Project

Make sure there is a `uv.lock` and `pyproject.toml` file in the setup directory. Then, run:

```console
uv sync
```

This will create a virtual environment in setup/.venv. Before running any code in this repo, make sure to run (from root):

```console
source setup/.venv/bin/activate
```

## Installing to alternate locations

If you don't want to have the virtual environment here, but a different location, then first create the venv elsewhere and activate it:
```console
uv venv /path/to/venv --python=3.XX.XX
source /path/to/venv/bin/activate
```
Then, come back to the setup directory and run:
```console
uv sync --active
```

If you do this, remember to add `/path/to/venv/` to the [configs file](../configs/private_vars.yaml)

Alternatively, you can create a simlink to the env path with (run this in the setup directory, not root):
```
ln -s /path/to/venv .venv 
```
Then, you can avoid the --active flag and do not need to alter the configs file. Install with:
```console
uv sync
```


# Creating New UV Project

If you want to create a new project, first cd into the [setup](../setup) directory and then run:
```console
uv init
rm main.py
```
We never use this main file, so delete it. This will create a project with the name `setup`. To change the project name, edit the `pyproject.toml` file. Also, make sure to add the exact python version to `pyproject.toml`, i.e. make it `==` not `>=` and then open `.python-version` and add the exact python version there as well.

You can now add pip packages with:
```console
uv add click huggingface_hub seaborn
```
These three packages are required for this codebase to run, but you can remove the `huggingface_hub` dependancy by removing [`sync_data.py`](../sync_data.py) and remove the `seaborn` dependancy by removing [Plotter](../utils/plot_handling.py).

This creates a uv.lock file, that must be committed to version control. This will also create the environment. Source it with (from root):


```console
source setup/.venv/bin/activate
```
