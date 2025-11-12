from poke_env import PokemonRedEmulator

if __name__ == "__main__":
    env = PokemonRedEmulator(parameters=None, headless=False)
    env._human_step_play()