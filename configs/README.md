## General Philosophy

You should put all private variables in the private_vars yaml (currently storage_dir is mandatory), the remaining can go into any .yaml file in this directory and will be read as parameters. Once you've defined the parameters, run the following command to create a config.env file in the configs folder. 

```bash
python configs/create_env_file.py
```

You can source this file to use config vars in bash scripts. 

You can use this README to list all relevant parameters and what they do.
