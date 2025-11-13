<p align="center">
  <picture>
    <img alt="Pokemon Environments" src="assets/blackicon.png"   width="500" height="400" style="max-width: 100%;">
  </picture>
  <br/>
  <br/>
</p>

<h2 align="center">
    <p>Gym Environments and Testbeds for Pokemon Red</p>
</h2>

<p align="center">
    <a href="https://github.com/DhananjayAshok/PokemonEnvironments/blob/main/LICENSE"><img alt="GitHub" src="https://img.shields.io/badge/license-MIT-blue"></a>
    <a href="https://dhananjayashok.github.io/"><img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/docs/transformers/index.svg?down_color=red&down_message=offline&up_message=online"></a>
</p>



This repo is under development

## Installation

The installation consists of four steps:
1. Environment Setup
2. ROM Setup
3. (Optional) Storage Directory Configuration
4. Final test

#### Environment Setup
Create and activate a virtual environment with [uv](https://docs.astral.sh/uv/), a fast Rust-based Python package and project manager.



```bash
uv venv /path/to/env
```
Then, source the environment
* On Windows:
```powershell
/path/to/env/Scripts/Activate
```
* On Linux:
```bash
source /path/to/env/bin/activate
```

Then, clone the repo and set up dependencies:
```
git clone https://github.com/DhananjayAshok/PokemonEnvironments
cd PokemonEnvironments
uv sync --active
```
#### ROM Setup

Next, you must legally acquire a ROM for Pokemon Red from [Nintendo](https://www.nintendo.com/en-gb/Games/Game-Boy/Pokemon-Red-Version-266109.html). We discourage any attempts to use this repository with unofficialy downloaded ROMs. 

Once you have a ROM (`.gb` file), place it in `PokemonEnvironments/rom_data/pokemon_red/PokemonRed.gb`. If you place it elsewhere, make sure to update the [config file](configs/pokemon_red_vars.yaml)

#### Storage Directory Configuration
By default, this project assumes that you can store emulator outputs (logs, screenshots, video captures etc.) in paths under the root directory of the project. If you wish to set a different location for storage, edit the appropriate configuration setting in the [config file](configs/private_vars.yaml).

#### Test

Check that setup is fine by running:
```bash
python tests/demo.py
```
This should open up a GameBoy window where you can play the Pokemon Red game. 


## Quickstart


## Core Features


## Citation

```bibtex
```
