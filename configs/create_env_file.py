"""
Run this script to read all the config files in the configs folder and create a .env file under configs. This env file can be sourced in to bash scripts to load all the parameters as environment variables.

Edit this file to change what keys get written to the .env file.
"""
import os
import yaml


def flatten_dict(d, parent_key='', sep='_'):
    """Flatten nested dictionaries with keys joined by sep."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def main():
    # check that configs directory exists and throw error if not
    if not os.path.exists("configs"):
        raise FileNotFoundError("configs directory not found. Please run this script from the root of the project.")

    yaml_files = [f for f in os.listdir("configs") if f.endswith(".yaml")]
    if not yaml_files:
        raise FileNotFoundError("No YAML files found in configs directory.")

    all_params = {}
    for yaml_file in yaml_files:
        with open(os.path.join("configs", yaml_file), 'r') as f:
            data = yaml.safe_load(f)
            all_params.update(data)
    fail_keys = []
    for key in all_params:
        if all_params[key] == "PLACEHOLDER":
            fail_keys.append(key)
        if key == "env_dir" and not os.path.exists(all_params[key]+ "/bin"):
            raise ValueError(f"Config variable {key} must be set to a virtual environment directory (but could not find {all_params[key]}/bin. Check that the path is correct.")
    if fail_keys:
        raise ValueError(f"The following keys have the value 'PLACEHOLDER'. Please set them in the config files before creating the env file: {fail_keys}")

    # Flatten in case of nested keys
    flat_data = flatten_dict(all_params)
    env_file = os.path.join("configs", "config.env")

    # Write to config.env
    with open(env_file, 'w') as f:
        for key, value in flat_data.items():
            # Ensure proper shell escaping for strings with spaces or special chars
            if isinstance(value, str):
                val_str = value.replace('"', '\\"')
                f.write(f'export {key}="{val_str}"\n')
            else:
                f.write(f'export {key}={value}\n')

    print(f"✅ Wrote {len(flat_data)} variables to {env_file}")
    print("Printing the env file contents below, check that everything looks correct:")
    with open(env_file, 'r') as f:
        print(f.read())

if __name__ == "__main__":
    main()
