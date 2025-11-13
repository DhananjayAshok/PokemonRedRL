from poke_env import BasicPokemonRedEmulator
import click


@click.command()
@click.option("--save_path", type=str, default="tmp.state", help="Path to save the .state file")
def main(save_path):
    env = BasicPokemonRedEmulator(parameters=None, headless=False)
    env._sav_to_state(save_path=save_path)

if __name__ == "__main__":
    main()