import uuid
import json
from pathlib import Path
from utils import log_warn, log_info, log_error, load_parameters
import numpy as np
from skimage.transform import downscale_local_mean
import matplotlib.pyplot as plt
from pyboy import PyBoy
#from pyboy.logger import log_level
import mediapy as media
from einops import repeat

from gymnasium import Env, spaces
from pyboy.utils import WindowEvent


event_flags_start = 0xD747
event_flags_end = 0xD87E # expand for SS Anne # old - 0xD7F6 
museum_ticket = (0xD754, 0)



import os

# TODO: Consider making cleaner
MAP_PATH = os.path.join(os.path.dirname(__file__), "map_data.json")
PAD = 20
GLOBAL_MAP_SHAPE = (444 + PAD * 2, 436 + PAD * 2)
MAP_ROW_OFFSET = PAD
MAP_COL_OFFSET = PAD

with open(MAP_PATH) as map_data:
    MAP_DATA = json.load(map_data)["regions"]
MAP_DATA = {int(e["id"]): e for e in MAP_DATA}

# Handle KeyErrors
def local_to_global(r: int, c: int, map_n: int):
    try:
        (
            map_x,
            map_y,
        ) = MAP_DATA[map_n]["coordinates"]
        gy = r + map_y + MAP_ROW_OFFSET
        gx = c + map_x + MAP_COL_OFFSET
        if 0 <= gy < GLOBAL_MAP_SHAPE[0] and 0 <= gx < GLOBAL_MAP_SHAPE[1]:
            return gy, gx
        print(f"coord out of bounds! global: ({gx}, {gy}) game: ({r}, {c}, {map_n})")
        return GLOBAL_MAP_SHAPE[0] // 2, GLOBAL_MAP_SHAPE[1] // 2
    except KeyError:
        print(f"Map id {map_n} not found in map_data.json.")
        return GLOBAL_MAP_SHAPE[0] // 2, GLOBAL_MAP_SHAPE[1] // 2

class GameInfo:
    def __init__(self, parameters):
        self.data = {}
        self.parameters = parameters
        
    
    def reset(self):
        self.data = {}
        # Thoughts of values here:
        """
        self.levels_satisfied = False
        self.base_explore = 0
        self.max_opponent_level = 0
        self.max_event_rew = 0
        self.max_level_rew = 0
        self.last_health = 1
        self.total_healing_rew = 0
        self.died_count = 0
        self.party_size = 0
        self.step_count = 0
        """
        

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

    

class RedGymEnv(Env):
    def __init__(self, parameters=None, **kwargs):
        parameters = load_parameters(parameters)
        kwargs.update(parameters)
        config = kwargs
        # assert the following must exist in configs:
        required_keys = [
            "gb_path",
            "session_path",
            "save_final_state",
            "headless",
            "init_state",
            "action_freq",
            "press_step",
            "max_steps",
            "save_video",
            "fast_video"
        ]
        for key in required_keys:
            if key not in config:
                log_error(f"Missing required config key: {key}")
        self.game_info = GameInfo(parameters)
        self.s_path = config["session_path"]
        self.save_final_state = config["save_final_state"]
        self.headless = config["headless"]
        self.init_state = config["init_state"]
        self.act_freq = config["action_freq"]
        self.press_step = config["press_step"]
        self.max_steps = config["max_steps"]
        self.save_video = config["save_video"]
        self.fast_video = config["fast_video"]
        self.frame_stacks = 3
        self.reward_scale = (
            1 if "reward_scale" not in config else config["reward_scale"]
        )
        self.instance_id = (
            str(uuid.uuid4())[:8]
            if "instance_id" not in config
            else config["instance_id"]
        )
        self.s_path.mkdir(exist_ok=True)
        self.full_frame_writer = None
        self.model_frame_writer = None
        self.reset_count = 0
        self.all_runs = []

        self.essential_map_locations = {
            v:i for i,v in enumerate([
                40, 0, 12, 1, 13, 51, 2, 54, 14, 59, 60, 61, 15, 3, 65
            ])
        }

        # Set this in SOME subclasses
        self.metadata = {"render.modes": []}

        self.valid_actions = [
            WindowEvent.PRESS_ARROW_DOWN,
            WindowEvent.PRESS_ARROW_LEFT,
            WindowEvent.PRESS_ARROW_RIGHT,
            WindowEvent.PRESS_ARROW_UP,
            WindowEvent.PRESS_BUTTON_A,
            WindowEvent.PRESS_BUTTON_B,
            WindowEvent.PRESS_BUTTON_START,
        ]

        self.release_actions = [
            WindowEvent.RELEASE_ARROW_DOWN,
            WindowEvent.RELEASE_ARROW_LEFT,
            WindowEvent.RELEASE_ARROW_RIGHT,
            WindowEvent.RELEASE_ARROW_UP,
            WindowEvent.RELEASE_BUTTON_A,
            WindowEvent.RELEASE_BUTTON_B,
            WindowEvent.RELEASE_BUTTON_START
        ]

        self.output_shape = (72, 80, self.frame_stacks)

        # Set these in ALL subclasses
        self.action_space = spaces.Discrete(len(self.valid_actions))
        
        self.observation_space = spaces.Dict({}) # Empty observation space. Will be accessed via other means. (Does not fit into Box Space)

        head = "null" if config["headless"] else "SDL2"

        self.pyboy = PyBoy(
            config["gb_path"],
            window=head,
        )

        #self.screen = self.pyboy.botsupport_manager().screen()

        if not config["headless"]:
            self.pyboy.set_emulation_speed(6)

    def reset(self, seed=None, options={}):
        self.seed = seed
        # restart game, skipping to init_state 
        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)

        self.game_info.reset()

        self.reset_count += 1
        return {}, {}

    def render(self, reduce_res=True):
        game_pixels_render = self.pyboy.screen.ndarray[:,:,0:1]  # (144, 160, 3)
        if reduce_res:
            game_pixels_render = (
                downscale_local_mean(game_pixels_render, (2,2,1))
            ).astype(np.uint8)
        return game_pixels_render
    
    def _get_obs(self):        
        observation = {}
        return observation

    def step(self, action):
        if self.save_video and self.step_count == 0:
            self.start_video()

        self.run_action_on_emulator(action)

        step_limit_reached = self.check_if_done()

        obs = self._get_obs()

        self.step_count += 1

        return obs, 0, False, step_limit_reached, {}
    
    def run_action_on_emulator(self, action):
        # press button then release after some steps
        self.pyboy.send_input(self.valid_actions[action])
        # disable rendering when we don't need it
        render_screen = self.save_video or not self.headless
        press_step = self.press_step
        self.pyboy.tick(press_step, render_screen)
        self.pyboy.send_input(self.release_actions[action])
        self.pyboy.tick(self.act_freq - press_step - 1, render_screen)
        self.pyboy.tick(1, True)
        if self.save_video and self.fast_video:
            self.add_video_frame()

    def start_video(self):
        # TODO: Manage paths better
        if self.full_frame_writer is not None:
            self.full_frame_writer.close()
        if self.model_frame_writer is not None:
            self.model_frame_writer.close()

        base_dir = self.s_path / Path("rollouts")
        base_dir.mkdir(exist_ok=True)
        full_name = Path(
            f"full_reset_{self.reset_count}_id{self.instance_id}"
        ).with_suffix(".mp4")
        model_name = Path(
            f"model_reset_{self.reset_count}_id{self.instance_id}"
        ).with_suffix(".mp4")
        self.full_frame_writer = media.VideoWriter(
            base_dir / full_name, (144, 160), fps=60, input_format="gray"
        )
        self.full_frame_writer.__enter__()
        self.model_frame_writer = media.VideoWriter(
            base_dir / model_name, self.output_shape[:2], fps=60, input_format="gray"
        )
        self.model_frame_writer.__enter__()

    def add_video_frame(self):
        self.full_frame_writer.add_image(
            self.render(reduce_res=False)[:,:,0]
        )
        self.model_frame_writer.add_image(
            self.render(reduce_res=True)[:,:,0]
        )
            

    def check_if_done(self):
        done = self.step_count >= self.max_steps - 1
        return done

    def save_and_print_info(self, done, obs):
        if self.step_count % 50 == 0:
            plt.imsave(
                self.s_path / Path(f"curframe_{self.instance_id}.jpeg"),
                self.render(reduce_res=False)[:,:, 0],
            )

        if self.save_final_state:
            fs_path = self.s_path / Path("final_states")
            fs_path.mkdir(exist_ok=True)
            plt.imsave(
                fs_path
                / Path(
                    f"frame_r{self.total_reward:.4f}_{self.reset_count}_full.jpeg"
                ),
                self.render(reduce_res=False)[:,:, 0],
            )

        if self.save_video and done:
            self.full_frame_writer.close()
            self.model_frame_writer.close()