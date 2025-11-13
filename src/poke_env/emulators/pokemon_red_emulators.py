# Borrowing heavily from https://github.com/PWhiddy/PokemonRedExperiments/ (v2)
# adapted from https://github.com/thatguy11325/pokemonred_puffer/blob/main/pokemonred_puffer/global_map.py


from poke_env.utils import log_warn, log_info, log_error, load_parameters
from poke_env.emulators.emulator import Emulator, GameStateParser

import os

import json
import numpy as np
from bidict import bidict


event_flags_start = 0xD747
event_flags_end = 0xD87E # expand for SS Anne # old - 0xD7F6 
museum_ticket = (0xD754, 0)


class GameInfo:
    def __init__(self, parameters):
        self.data = {}
        self.parameters = parameters
        
    
    def reset(self):
        self.data = {}
        

class PokemonRedGameStateParser(GameStateParser):
    """
    Reads from memory addresses to form the state: https://github.com/pret/pokered/blob/symbols/pokered.sym
    """
    PAD = 20
    GLOBAL_MAP_SHAPE = (444 + PAD * 2, 436 + PAD * 2)
    MAP_ROW_OFFSET = PAD
    MAP_COL_OFFSET = PAD
    def __init__(self, pyboy, parameters):
        super().__init__(pyboy, parameters)
        # load event names (parsed from https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm)
        events_location = self.parameters["pokemon_red_events_path"]
        with open(events_location) as f:
            event_slots = json.load(f)
        event_slots = event_slots
        event_names = {v: k for k, v in event_slots.items() if not v[0].isdigit()}
        beat_opponent_events = bidict()
        def _pop(d, keys):
            for key in keys:
                d.pop(key, None)        
        pop_queue = []
        for name, slot in event_names.items():
            if name.startswith("Beat "):
                beat_opponent_events[name.replace("Beat ", "")] = slot
                pop_queue.append(name)
        _pop(event_names, pop_queue)
        self.beat_opponent_events = beat_opponent_events
        tms_obtained_events = bidict()
        pop_queue = []
        for name, slot in event_names.items():
            if name.startswith("Got Tm"):
                tms_obtained_events[name.replace("Got ", "").strip()] = slot
                pop_queue.append(name)
        _pop(event_names, pop_queue)
        self.tms_obtained_events = tms_obtained_events
        hm_obtained_events = bidict()
        pop_queue = []
        for name, slot in event_names.items():
            if name.startswith("Got Hm"):
                hm_obtained_events[name.replace("Got ", "").strip()] = slot
                pop_queue.append(name)
        _pop(event_names, pop_queue)
        self.hm_obtained_events = hm_obtained_events
        passed_badge_check_events = bidict()
        pop_queue = []
        for name, slot in event_names.items():
            if name.startswith("Passed ") and "badge" in name:
                passed_badge_check_events[name.replace("Passed ", "").replace(" Check", "").strip()] = slot
                pop_queue.append(name)
        _pop(event_names, pop_queue)
        self.passed_badge_check_events = passed_badge_check_events
        self.key_items_obtained_events = bidict()
        pop_queue = []
        for name, slot in event_names.items():
            if name.startswith("Got "):
                self.key_items_obtained_events[name.replace("Got ", "").strip()] = slot
                pop_queue.append(name)
        _pop(event_names, pop_queue)
        self.map_events = {"Cinnabar Gym": bidict(), "Victory Road": bidict(), "Silph Co": bidict(), "Seafoam Islands": bidict()}
        for name, slot in event_names.items():
            if name.startswith("Cinnabar Gym Gate") and name.endswith("Unlocked"):
                self.map_events["Cinnabar Gym"][name] = slot
                pop_queue.append(name)
            elif name.startswith("Victory Road") and "Boulder On" in name:
                self.map_events["Victory Road"][name] = slot
                pop_queue.append(name)
            elif name.startswith("Silph Co") and "Unlocked" in name:
                self.map_events["Silph Co"][name] = slot
                pop_queue.append(name)
            elif name.startswith("Seafoam"):
                self.map_events["Seafoam Islands"][name] = slot
                pop_queue.append(name)
        _pop(event_names, pop_queue)
        cutscene_events = bidict()
        cutscenes = ["Event 001", "Daisy Walking", "Pokemon Tower Rival On Left", "Seel Fan Boast", "Pikachu Fan Boast", "Lab Handing Over Fossil Mon", "Route22 Rival Wants Battle"] # my best guess, need to verify, Silph Co Receptionist At Desk? Autowalks?


        self.special_events = bidict(event_names)


        self.essential_map_locations = {
            v:i for i,v in enumerate([
                40, 0, 12, 1, 13, 51, 2, 54, 14, 59, 60, 61, 15, 3, 65
            ])
        }
        MAP_PATH = parameters["pokemon_red_map_data_path"]
        with open(MAP_PATH) as map_data:
            MAP_DATA = json.load(map_data)["regions"]
        self.MAP_DATA = {int(e["id"]): e for e in MAP_DATA}
        
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
            return self.local_to_global(y_pos, x_pos, map_n)
    
    def __repr__(self):
        return "PokemonRed"
    
    def parse_all(self):
        pass

    def parse_step(self):
        self.parsed_variables["local_coords"] = self.get_game_coords()
        self.parsed_variables["global_coords"] = self.get_global_coords()
        self.parsed_variables["in_battle"] = self.isinbattle()
        self.parsed_variables["party_size"] = self.get_party_size()
        self.parsed_variables["party_levels"] = self.read_party_levels()
        self.parsed_variables["badges"] = self.get_badges()
        self.parsed_variables["hp_fraction"] = self.read_hp_fraction()
        self.parsed_variables["event_flags"] = self.read_event_bits()
        self.parsed_variables["opponent_levels"] = self.get_opponent_levels()
        self.parsed_variables["player_level"] = self.get_levels()[0]
        self.parsed_variables["event_flags"] = self.read_event_bits()
        self.parsed_variables["opponent_levels"] = self.get_opponent_levels()



    
    
class BasicPokemonRedEmulator(Emulator):
    def __init__(self, parameters: dict = None, init_state=None, headless: bool = False, max_steps: int = None, save_video: bool = None, session_name: str = None, instance_id: str = None):
        parameters = load_parameters(parameters)
        if init_state is None:
            init_state = parameters["pokemon_red_default_state"]        
        gb_path = parameters["pokemon_red_gb_path"]
        game_state_parser_class = PokemonRedGameStateParser
        super().__init__(gb_path, game_state_parser_class, init_state, parameters, headless, max_steps, save_video, session_name, instance_id)

    def _get_obs(self):        
        observation = {}
        return observation
    
    def get_env_variant(self) -> str:
        """        
        Returns a string identifier for the particular environment variant being used.
        
        :return: string name identifier of the particular env e.g. PokemonRed
        """
        return "pokemon_red"
