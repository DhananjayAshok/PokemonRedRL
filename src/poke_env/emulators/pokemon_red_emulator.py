# Borrowing heavily from https://github.com/PWhiddy/PokemonRedExperiments/ (v2)
# adapted from https://github.com/thatguy11325/pokemonred_puffer/blob/main/pokemonred_puffer/global_map.py


from poke_env.utils import log_warn, log_info, log_error, load_parameters
from poke_env.emulators.emulator import Emulator

import json
import numpy as np
import os


event_flags_start = 0xD747
event_flags_end = 0xD87E # expand for SS Anne # old - 0xD7F6 
museum_ticket = (0xD754, 0)


class GameInfo:
    def __init__(self, parameters):
        self.data = {}
        self.parameters = parameters
        
    
    def reset(self):
        self.data = {}
        

class GameStateParser:
    def __init__(self, pyboy, parameters):
        self.pyboy = pyboy
        self.parameters = parameters
        # load event names (parsed from https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm)
        with open("events.json") as f:
            event_names = json.load(f)
        self.event_names = event_names
    
    def bit_count(self, bits):
        return bin(bits).count("1")    
    
    def read_m(self, addr):
        #return self.pyboy.get_memory_value(addr)
        return self.pyboy.memory[addr]

    def read_bit(self, addr, bit: int) -> bool:
        # add padding so zero will read '0b100000000' instead of '0b0'
        return bin(256 + self.read_m(addr))[-bit - 1] == "1"

    def read_event_bits(self):
        return [
            int(bit) for i in range(event_flags_start, event_flags_end) 
            for bit in f"{self.read_m(i):08b}"
        ]
        
    def get_game_coords(self):
        return (self.read_m(0xD362), self.read_m(0xD361), self.read_m(0xD35E))

    def read_hp_fraction(self):
        hp_sum = sum([
            self.read_hp(add)
            for add in [0xD16C, 0xD198, 0xD1C4, 0xD1F0, 0xD21C, 0xD248]
        ])
        max_hp_sum = sum([
            self.read_hp(add)
            for add in [0xD18D, 0xD1B9, 0xD1E5, 0xD211, 0xD23D, 0xD269]
        ])
        max_hp_sum = max(max_hp_sum, 1)
        return hp_sum / max_hp_sum

    def read_hp(self, start):
        return 256 * self.read_m(start) + self.read_m(start + 1)
        
    def idk_imp(self):
        self.base_event_flags = sum([
                self.bit_count(self.read_m(i))
                for i in range(event_flags_start, event_flags_end)
        ])
        self.pcount = self.read_m(0xD163)
        
    def get_levels(self):
        return [
            self.read_m(a) for a in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]
        ]
        
    def publish_game_state(self):
        return {}

    def get_opponent_levels(self):
        opp_base_level = 5
        opponent_levels = [self.read_m(a) for a in [0xD8C5, 0xD8F1, 0xD91D, 0xD949, 0xD975, 0xD9A1]]
        return [level - opp_base_level for level in opponent_levels] # TODO: confirm base level

    def get_badges(self):
        # or  self.bit_count(self.read_m(0xD356))
        return np.array([int(bit) for bit in f"{self.read_m(0xD356):08b}"], dtype=np.int8)
    
    def get_party_size(self):
        party_size = self.read_m(0xD163) # This is the number of active Pokemon in party I think. TODO: confirm it does not count fainted Pokemon
        return party_size
    
    def isinbattle(self):
        return self.read_m(0xD057) != 0

    def read_party(self):
        return [
            self.read_m(addr)
            for addr in [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]
        ]
        
    def read_party_levels(self):
        min_poke_level = 2 # I don't know how this fits in yet. Do we take offset?
        return [self.read_m(addr) for addr in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]]
    
    def get_global_coords(self):
            x_pos, y_pos, map_n = self.get_game_coords()
            return local_to_global(y_pos, x_pos, map_n)

    
    
class PokemonRedEmulator(Emulator):
    PAD = 20
    GLOBAL_MAP_SHAPE = (444 + PAD * 2, 436 + PAD * 2)
    MAP_ROW_OFFSET = PAD
    MAP_COL_OFFSET = PAD

    def __init__(self, parameters: dict, init_state=None, headless: bool = False, max_steps: int = None, save_video: bool = None, fast_video: bool = None, session_name: str = None, instance_id: str = None):
        assert parameters is not None, "You must provide a parameters dictionary to the Emulator"
        assert len(parameters) > 0, "The parameters dictionary cannot be empty"
        if init_state is None:
            init_state = parameters["pokemon_red_default_state"]        
        gb_path = parameters["pokemon_red_gb_path"]
        super().__init__(gb_path, init_state, parameters, headless, max_steps, save_video, fast_video, session_name, instance_id)
        self.essential_map_locations = {
            v:i for i,v in enumerate([
                40, 0, 12, 1, 13, 51, 2, 54, 14, 59, 60, 61, 15, 3, 65
            ])
        }
        MAP_PATH = parameters["pokemon_red_map_data_path"]
        with open(MAP_PATH) as map_data:
            MAP_DATA = json.load(map_data)["regions"]
        MAP_DATA = {int(e["id"]): e for e in MAP_DATA}
        
    # Handle KeyErrors. TODO: Figure out what this is doing. 
    def local_to_global(self, r: int, c: int, map_n: int):
        try:
            (
                map_x,
                map_y,
            ) = self.MAP_DATA[map_n]["coordinates"]
            gy = r + map_y + self.MAP_ROW_OFFSET
            gx = c + map_x + self.MAP_COL_OFFSET
            if 0 <= gy < self.GLOBAL_MAP_SHAPE[0] and 0 <= gx < self.GLOBAL_MAP_SHAPE[1]:
                return gy, gx
            print(f"coord out of bounds! global: ({gx}, {gy}) game: ({r}, {c}, {map_n})")
            return self.GLOBAL_MAP_SHAPE[0] // 2, self.GLOBAL_MAP_SHAPE[1] // 2
        except KeyError:
            print(f"Map id {map_n} not found in map_data.json.")
            return self.GLOBAL_MAP_SHAPE[0] // 2, self.GLOBAL_MAP_SHAPE[1] // 2
            
    def _get_obs(self):        
        observation = {}
        return observation
    
    def get_env_variant(self) -> str:
        """        
        Returns a string identifier for the particular environment variant being used.
        
        :return: string name identifier of the particular env e.g. PokemonRed
        """
        return "pokemon_red"
