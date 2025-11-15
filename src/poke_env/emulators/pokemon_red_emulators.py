# Borrowing heavily from https://github.com/PWhiddy/PokemonRedExperiments/ (v2)
# adapted from https://github.com/thatguy11325/pokemonred_puffer/blob/main/pokemonred_puffer/global_map.py


from poke_env.utils import log_warn, log_info, log_error, load_parameters
from poke_env.emulators.structs import Pokemon
from poke_env.emulators.emulator import Emulator, GameStateParser

from typing import Set, List
import os

import json
import numpy as np
from bidict import bidict


event_flags_start = 0xD747
event_flags_end = 0xD87E # expand for SS Anne # old - 0xD7F6 
museum_ticket = (0xD754, 0)
# TODO: Do you need these?



class ControlProcedures:
    # TODO: Must implement the following
    # Autobattler (simple, first check if there is an attacking move in any pokemons slots. If so and not active, switch into it. Then spam that attack)
    # This is meant to be used with nerf_opponents to allow simulation without fear of battles getting in the way. 
    # A* pathfinding towards a given coordinate. 
    # Menu: Open Items, Open Pokemon, 
    # Open Map, Exit Map
    # Throw Ball
    # Show inventory 
    # Use Specific Item (e.g. Antidote etc) (String Based) You must establish the mapping to game id and get that done
    # Use Item on Pokemon 
    # Run from Battle
    # Show other pokemon info
    # Check Pokemon Info
    # Switch to Pokemon
    # Maybe set up an OCR on the frame to catch some amount of string mapping (i.e. catch cant use that )
        # Then you can have OCR on a sign
    # Simple mappings of interact with NPC or sign. 
    # Move in direction (for a specified number of steps)
    # Try to Buy x Items
    # 
    pass
    # TODO: See if you can find a way to draw the local coordinate system on your map. To help the VLM specify pathfinding coordinates. 
"""
Action Spaces: 
Open World, No Window open:
    Move:
        Move(direction, steps): either returns success message or failure and early exit (could be wild pokemon, could be trainer, could be obstacle). 
        Move(x, y): Try to move towards this coordinate using A* algorithm. 
    Interact: Basically press A
    Inventory:
        Search(Item Name): return [not an item or the count of item in bag]
        List items: List all items in bag
        Use(Item Name):
            For each item, perhaps have an argument input (i.e. Use Potion on [Pokemon Name])
    Pokemon:
        List: List all Pokemon
        Check(Pokemon Name)
        Check_all
        SwitchFirst(Pokemon): Switches a new pokemon to the first slot in the party
    Fly: (If there is a fly pokemon in inventory)


Battle:
    Fight:
        Attack(name): either says attack not found or uses the attack
    Inventory:
        List
        Search(Item Name)
        Use(Item Name)
    Throw Ball(Ball Kind)
    Pokemon:
        List
        Check(Pokemon Name)
        Switch(Pokemon Name)
    Run

Conversation: (Hard?)
    Continue

Menu / Options: (HARD)
    Select specific option

Manual:
    All controls (other than Enter)    
"""





class GameInfo:
    def __init__(self, parameters):
        self.data = {}
        self.parameters = parameters
        
    
    def reset(self):
        self.data = {}
        
class Hacker:
    def __init__(self, pyboy):
        self.pyboy = pyboy

    def disable_wild_encounters(self):
        """
        TODO: 
        """
        self.pyboy.memory[0xd887] = 0 # grass rate
        self.pyboy.memory[0xd8a4] = 0

    def nerf_opponent_trainers(self):
        """
        """
        pass


class PokemonRedGameStateParser(GameStateParser):
    """
    Reads from memory addresses to form the state: https://github.com/pret/pokered/blob/symbols/pokered.sym and https://datacrystal.tcrf.net/wiki/Pok%C3%A9mon_Red_and_Blue/RAM_map    
    """
    _PAD = 20
    _GLOBAL_MAP_SHAPE = (444 + _PAD * 2, 436 + _PAD * 2)
    _MAP_ROW_OFFSET = _PAD
    _MAP_COL_OFFSET = _PAD
    def __init__(self, pyboy, parameters):
        """
        Initializes the Pokemon Red game state parser.

        Args: 
            pyboy: An instance of the PyBoy emulator.
            parameters: A dictionary of parameters for configuration.
        """
        super().__init__(pyboy, parameters)
        # load event names (parsed from https://github.com/pret/pokered/blob/91dc3c9f9c8fd529bb6e8307b58b96efa0bec67e/constants/event_constants.asm)
        events_location = self._parameters["pokemon_red_events_path"]
        with open(events_location) as f:
            event_slots = json.load(f)
        event_slots = event_slots
        event_names = {v: k for k, v in event_slots.items() if not v[0].isdigit()}
        beat_opponent_events = bidict()
        def _pop(d, keys):
            for key in keys:
                if key in d:
                    d.pop(key, None)        
        pop_queue = []
        for name, slot in event_names.items():
            if name.startswith("Beat "):
                beat_opponent_events[name.replace("Beat ", "")] = slot
                pop_queue.append(name)
        _pop(event_names, pop_queue)
        self.defeated_opponent_events = beat_opponent_events
        """Events related to beating specific opponents. E.g. Beat Brock"""
        tms_obtained_events = bidict()
        pop_queue = []
        for name, slot in event_names.items():
            if name.startswith("Got Tm"):
                tms_obtained_events[name.replace("Got ", "").strip()] = slot
                pop_queue.append(name)
        _pop(event_names, pop_queue)
        self.tms_obtained_events = tms_obtained_events
        """Events related to obtaining specific TMs. E.g. Got Tm01"""
        hm_obtained_events = bidict()
        pop_queue = []
        for name, slot in event_names.items():
            if name.startswith("Got Hm"):
                hm_obtained_events[name.replace("Got ", "").strip()] = slot
                pop_queue.append(name)
        _pop(event_names, pop_queue)
        self.hm_obtained_events = hm_obtained_events
        """Events related to obtaining specific HMs. E.g. Got Hm01"""
        passed_badge_check_events = bidict()
        pop_queue = []
        for name, slot in event_names.items():
            if name.startswith("Passed ") and "badge" in name:
                passed_badge_check_events[name.replace("Passed ", "").replace(" Check", "").strip()] = slot
                pop_queue.append(name)
        _pop(event_names, pop_queue)
        self.passed_badge_check_events = passed_badge_check_events
        """Events related to passing badge checks. E.g. Passed Boulder badge check. These will only be relevant to enter Victory Road."""
        self.key_items_obtained_events = bidict()
        """Events related to obtaining key items. E.g. Got Bicycle"""
        pop_queue = []
        for name, slot in event_names.items():
            if name.startswith("Got "):
                self.key_items_obtained_events[name.replace("Got ", "").strip()] = slot
                pop_queue.append(name)
        _pop(event_names, pop_queue)
        self.map_events = {"Cinnabar Gym": bidict(), "Victory Road": bidict(), "Silph Co": bidict(), "Seafoam Islands": bidict()}
        """Events related to specific map events like unlocking gates or moving boulders."""
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
        self.cutscene_events = bidict()
        """ Flags for cutscene based events. TODO: Check if these are actually flags etc. """

        cutscenes = ["Event 001", "Daisy Walking", "Pokemon Tower Rival On Left", "Seel Fan Boast", "Pikachu Fan Boast", "Lab Handing Over Fossil Mon", "Route22 Rival Wants Battle"] # my best guess, need to verify, Silph Co Receptionist At Desk? Autowalks?
        pop_queue = []
        for name, slot in event_names.items():
            if name in cutscenes:
                self.cutscene_events[name] = slot
                pop_queue.append(name)
        _pop(event_names, pop_queue)
        self.special_events = bidict(event_names)
        """ All other events not categorized elsewhere."""


        self.essential_map_locations = {
            v:i for i,v in enumerate([
                40, 0, 12, 1, 13, 51, 2, 54, 14, 59, 60, 61, 15, 3, 65
            ])
        } # TODO: Do I need this?
        MAP_PATH = parameters["pokemon_red_map_data_path"]
        with open(MAP_PATH) as map_data:
            MAP_DATA = json.load(map_data)["regions"]
        self._MAP_DATA = {int(e["id"]): e for e in MAP_DATA}
        
    def local_to_global(self, r: int, c: int, map_n: int) -> tuple[int, int]:
        """
        Converts local map coordinates to global map coordinates.
        Args:
            r (int): Local row coordinate.
            c (int): Local column coordinate.
            map_n (int): Map identifier.
        Returns:
            (int, int): Global (row, column) coordinates.
        """
        try:
            (
                map_x,
                map_y,
            ) = self._MAP_DATA[map_n]["coordinates"]
            gy = r + map_y + self._MAP_ROW_OFFSET
            gx = c + map_x + self._MAP_COL_OFFSET
            if 0 <= gy < self._GLOBAL_MAP_SHAPE[0] and 0 <= gx < self._GLOBAL_MAP_SHAPE[1]:
                return gy, gx
            print(f"coord out of bounds! global: ({gx}, {gy}) game: ({r}, {c}, {map_n})")
            return self._GLOBAL_MAP_SHAPE[0] // 2, self._GLOBAL_MAP_SHAPE[1] // 2
        except KeyError:
            print(f"Map id {map_n} not found in map_data.json.")
            return self._GLOBAL_MAP_SHAPE[0] // 2, self._GLOBAL_MAP_SHAPE[1] // 2

    def bit_count(self, bits: int) -> int:
        """
        Counts the number of set bits (1s) in the given integer.
        Args:
            bits (int): The integer to count set bits in.
        Returns:
            int: The number of set bits.
        """
        return bin(bits).count("1")    
    
    def read_m(self, addr: bytes) -> int:
        """
        Reads a byte from the specified memory address.
        Args:
            addr (int): The memory address to read from.
        Returns:
            int: The byte value at the specified memory address.
        """
        #return self.pyboy.get_memory_value(addr)
        return self._pyboy.memory[addr]

    def read_bits(self, addr) -> str:
        """
        Reads a memory address and returns the result as a binary string. Adds padding so that reading bit 0 works correctly. 
        Args:
            addr (int): The memory address to read from.
        Returns:
            str: The binary string representation of the byte at the specified memory address.
        """
        # add padding so zero will read '0b100000000' instead of '0b0'
        return bin(256 + self.read_m(addr))

    def read_bit(self, addr, bit: int) -> bool:
        """
        Reads a specific bit from a memory address.
        Args:
            addr (int): The memory address to read from.
            bit (int): The bit position to read (0-7).
        Returns:
            bool: True if the bit is set (1), False otherwise.
        """
        # add padding so zero will read '0b100000000' instead of '0b0'
        return self.read_bits(addr)[-bit - 1] == "1"
    
    def read_m_bit(self, addr_bit: str) -> bool:
        """
        Reads a specific addr-bit string from a memory address. 
        Args:
            addr_bit (str): The - concatenation of a memory address and the bit position (e.g. '0xD87D-5')
        Returns:
            bool: True if the bit at that memory address is set (1), False otherwise
        """
        if "-" not in addr_bit:
            log_error(f"Incorrect format addr_bit: {addr_bit}", self._parameters)
        addr, bit = addr_bit.split("-")        
        flag = False
        try:
            addr = eval(addr)
        except:
            flag = True
        if flag:
            log_error(f"Could not eval byte string: {addr}. Check format", self._parameters)
        if not bit.isdigit():
            log_error(f"bit {bit} is not digit", self._parameters)
        bit = int(bit)
        return self.read_bit(addr, bit)

    def read_consecutive_m(self, start_addr, n_slots):
        # TODO: 
        pass

    def _get_items(self, item_dict):
        items = set()
        for item_name, slot in item_dict.items():
            if self.read_m_bit(slot):
                items.add(item_name)
        return items

    def get_opponents_defeated(self) -> Set[str]:
        """
        Returns a set of all defeated opponents.
        
        """
        return self._get_items(self.defeated_opponent_events)
    
    def get_all_opponents(self) -> Set[str]:
        """
        Returns a set of all possible opponents. 
        """
        return set(self.defeated_opponent_events.keys())
    
    # Same for TMs and HMs and Key Items
    def get_obtained_tms(self) -> Set[str]:
        """
        Returns a set of all obtained TMs.
        """
        return self._get_items(self.tms_obtained_events)

    def get_all_tms(self) -> Set[str]:
        """
        Returns a set of all TMs.
        """
        return set(self.tms_obtained_events.keys())
    
    def get_obtained_hms(self) -> Set[str]:
        """
        Returns a set of all obtained HMs.
        """
        return self._get_items(self.hm_obtained_events)
    
    def get_all_hms(self) -> Set[str]:
        """
        Returns a set of all HMs.
        """
        return set(self.hm_obtained_events.keys())

    def get_obtained_key_items(self) -> Set[str]:
        """
        Returns a set of all obtained key items.        
        """
        return self._get_items(self.key_items_obtained_events)

    def get_all_key_items(self) -> Set[str]:
        """
        Returns a set of all key items.
        """
        return set(self.key_items_obtained_events.keys())

    def get_cutscene_events(self) -> Set[str]:
        """
        Returns a set of all triggered cutscene events.
        """
        return self._get_items(self.cutscene_events)
    
    def get_all_cutscene_events(self) -> Set[str]:
        """
        Returns a set of all cutscene events.
        """
        return set(self.cutscene_events.keys())
        
    def get_local_coords(self) -> tuple[int, int, int]:
        """
        Gets the local game coordinates (x, y, map number).
        Returns:
            (int, int, int): Tuple containing (x, y, map number).
        """
        return (self.read_m(0xD362), self.read_m(0xD361), self.read_m(0xD35E))

    def get_global_coords(self):
        """
        Gets the global coordinates of the player.
        Returns:
            (int, int): Tuple containing (global y, global x) coordinates.
        """
        x_pos, y_pos, map_n = self.get_local_coords()
        return self.local_to_global(y_pos, x_pos, map_n)

    def get_badges(self) -> np.array:
        """
        Gets the player's badges as a binary array.
        Returns:
            np.array: Array of 8 binary values representing whether the player has obtained each of the badges.
        """
        # or  self.bit_count(self.read_m(0xD356))
        return np.array([int(bit) for bit in f"{self.read_m(0xD356):08b}"], dtype=np.int8)
    
    def get_party_size(self) -> int:
        """
        TODO: confirm
        Returns:
            int: The number of Pokemon in the player's party.
        """
        party_size = self.read_m(0xD163) # This is the number of active Pokemon in party I think. TODO: confirm it does not count fainted Pokemon
        return party_size
    
    def isinbattle(self) -> bool:
        """
        Checks if the player is currently in a battle.
        Returns:
            bool: True if in battle, False otherwise.
        """
        return self.read_m(0xD057) != 0

    def read_party(self):
        """
        #TODO: understand
        """
        return [
            self.read_m(addr)
            for addr in [0xD164, 0xD165, 0xD166, 0xD167, 0xD168, 0xD169]
        ]
        
    def read_party_levels(self):
        """
        #TODO: understand
        """
        min_poke_level = 2 # I don't know how this fits in yet. Do we take offset?
        return [self.read_m(addr) for addr in [0xD18C, 0xD1B8, 0xD1E4, 0xD210, 0xD23C, 0xD268]]
    
    def _get_current_menu_item(self) -> int:
        """
        Returns the integer index of the current item that the player selection cursor is hovering over in a menu.

        Returns:
            int: The index of the current (OR PREVIOUSLY CLOSED) menu item. 
            Cannot be used as an indicator for menu_open
        """
        return self.read_m(0xCC26)
    
    def get_n_menu_items(self) -> int:
        """
        Returns the integer number of items in the current menu.

        Returns:
            int: The number of items in the current menu.
            Cannot be used as an indicator for menu_open (i.e. never returns 0)
        """
        return self.read_m(0xCC28)

    # TODO Section:
    def get_player_pokemon(self, party_id: int) -> Pokemon:
        """
        Returns a Pokemon object representing the player's Pokemon at the specified party index.

        Args:
            party_id (int): The index of the Pokemon in the player's party (0-5).

        Returns:
            Pokemon: The Pokemon object representing the player's Pokemon.
        """
        # The pokemon initializer takes:
        # class Pokemon: __init__(self, species, type1, type2, level, current_hp, status, hp_ev, attack_ev, defense_ev, speed_ev, special_ev, attack_defense_iv, speed_special_iv, max_hp, attack, defense, speed, special, move1, move2, move3, move4, move_1_pp, move_2_pp, move_3_pp, move_4_pp)
        if party_id == 0:
            # set the relevant variables here
            pass
        elif party_id == 1:
            # set
            pass
        elif party_id == 2:
            # set
            pass
        elif party_id == 3:
            pass
        elif party_id == 4:
            pass
        elif party_id == 5:
            pass
        elif party_id == 6:
            pass
        else:
            log_error(f"Only 6 pokemon max", self._parameters)
        return Pokemon(self.read_m(species_addr), self.read_m(type1_addr), self.read_m(type2_addr), 
                       self.read_m(level_addr), self.read_m(current_hp_addr), self.read_m(status_addr), self.read_m(hp_ev_addr), self.read_m(attack_ev_addr), self.read_m(defense_ev_addr), self.read_m(speed_ev_addr), self.read_m(special_ev_addr), self.read_m(attack_defense_iv_addr), self.read_m(speed_special_iv_addr), self.read_m(max_hp_addr), self.read_m(attack_addr), self.read_m(defense_addr), self.read_m(speed_addr), self.read_m(special_addr), self.read_m(move1_addr), self.read_m(move2_addr), self.read_m(move3_addr), self.read_m(move4_addr), self.read_m(move_1_pp_addr), self.read_m(move_2_pp_addr), self.read_m(move_3_pp_addr), self.read_m(move_4_pp_addr))

    def get_opponent_pokemon(self, party_id: int) -> Pokemon:
        """
        Returns a Pokemon object representing the enemy's Pokemon at the specified index.

        Args:
            enemy_id (int): The index of the enemy Pokemon (0 for single battles, 0 and 1 for double battles).
        Returns:
            Pokemon: The Pokemon object representing the enemy's Pokemon.
        """
                # The pokemon initializer takes:
        # class Pokemon: __init__(self, species, type1, type2, level, current_hp, status, hp_ev, attack_ev, defense_ev, speed_ev, special_ev, attack_defense_iv, speed_special_iv, max_hp, attack, defense, speed, special, move1, move2, move3, move4, move_1_pp, move_2_pp, move_3_pp, move_4_pp)
        if party_id == 0:
            # set the relevant variables here
            pass
        elif party_id == 1:
            # set
            pass
        elif party_id == 2:
            # set
            pass
        elif party_id == 3:
            pass
        elif party_id == 4:
            pass
        elif party_id == 5:
            pass
        elif party_id == 6:
            pass
        else:
            log_error(f"Only 6 pokemon max", self._parameters)
        return Pokemon(self.read_m(species_addr), self.read_m(type1_addr), self.read_m(type2_addr), 
                       self.read_m(level_addr), self.read_m(current_hp_addr), self.read_m(status_addr), self.read_m(hp_ev_addr), self.read_m(attack_ev_addr), self.read_m(defense_ev_addr), self.read_m(speed_ev_addr), self.read_m(special_ev_addr), self.read_m(attack_defense_iv_addr), self.read_m(speed_special_iv_addr), self.read_m(max_hp_addr), self.read_m(attack_addr), self.read_m(defense_addr), self.read_m(speed_addr), self.read_m(special_addr), self.read_m(move1_addr), self.read_m(move2_addr), self.read_m(move3_addr), self.read_m(move4_addr), self.read_m(move_1_pp_addr), self.read_m(move_2_pp_addr), self.read_m(move_3_pp_addr), self.read_m(move_4_pp_addr))

    def get_all_player_pokemon(self) -> List[Pokemon]:
        """
        """
        pass

    def get_all_opponent_pokemon(self) -> List[Pokemon]:
        pass

    def get_species_caught(self) -> List[str]:
        # Go through PokeDex, find all caught species
        pass

    def get_species_seen(self) -> List[str]:
        # Go through PokeDex, find all seen species
        pass


    def __repr__(self):
        return "PokemonRed"
    
    def parse_all(self):
        pass

    def parse_step(self):
        self.parsed_variables["local_coords"] = self.get_local_coords()
        self.parsed_variables["global_coords"] = self.get_global_coords()
        self.parsed_variables["in_battle"] = self.isinbattle()
        self.parsed_variables["party_size"] = self.get_party_size()
        self.parsed_variables["party_levels"] = self.read_party_levels()
        self.parsed_variables["badges"] = self.get_badges()
        self.parsed_variables["opponents_defeated"] = self.get_opponents_defeated()
        self.parsed_variables["tms_obtained"] = self.get_obtained_tms()
        self.parsed_variables["hms_obtained"] = self.get_obtained_hms()
        self.parsed_variables["key_items_obtained"] = self.get_obtained_key_items()
        self.parsed_variables["cutscene_events_triggered"] = self.get_cutscene_events()
        self.parsed_variables["menu_item_idx"] = self._get_current_menu_item()
        self.parsed_variables["menu_length"] = self.get_n_menu_items()
        test_bit_strings = ["d8a6"] # enemy p1 hp 
        # d12e maybe can still do something
        test_bits = []
        for bit_string in test_bit_strings:
            test_bits.append(f"0x{bit_string}")
        for test in test_bits:
            self.parsed_variables[f"test_bit_{test}"] = self.read_m(eval(test))
        #self._pyboy.memory[0xd8a6] = 1
        #self._pyboy.memory[0xD8C1] = 0
        #self._pyboy.memory[0xD8C2] = 0        
        #self._pyboy.memory[0xD8C3] = 0
        #self._pyboy.memory[0xD8C4] = 0
        



    

    
    
class BasicPokemonRedEmulator(Emulator):
    def __init__(self, parameters: dict = None, init_state=None, headless: bool = False, max_steps: int = None, save_video: bool = None, session_name: str = None, instance_id: str = None):
        parameters = load_parameters(parameters)
        if init_state is None:
            init_state = parameters["pokemon_red_default_state"]        
        gb_path = parameters["pokemon_red_gb_path"]
        game_state_parser_class = PokemonRedGameStateParser
        super().__init__(gb_path, game_state_parser_class, init_state, parameters, headless, max_steps, save_video, session_name, instance_id)
    
    def get_env_variant(self) -> str:
        """        
        Returns a string identifier for the particular environment variant being used.
        
        :return: string name identifier of the particular env e.g. PokemonRed
        """
        return "pokemon_red"
