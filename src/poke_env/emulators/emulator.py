from enum import Enum
import os
import uuid
from poke_env.utils import load_parameters, log_error, log_warn, file_makedir


import mediapy as media
from pyboy import PyBoy
from pyboy.utils import WindowEvent
from matplotlib import pyplot as plt
from skimage.transform import downscale_local_mean
import numpy as np


def allocate_new_session_name(parameters, session_path):
    storage_dir = parameters["storage_dir"]
    session_path = os.path.join(storage_dir, session_path)
    os.makedirs(session_path, exist_ok=True)
    existing_sessions = os.listdir(session_path)
    # session names are always in the form session_X where X is an integer starting from 0
    session_indices = [int(name.split("_")[-1]) for name in existing_sessions if name.startswith("session_") and name.split("_")[-1].isdigit()]
    if len(session_indices) == 0:
        new_index = 0
    else:
        new_index = max(session_indices) + 1
    new_session_name = f"session_{new_index}"
    return new_session_name

class LowLevelActions(Enum):
    PRESS_ARROW_DOWN = WindowEvent.PRESS_ARROW_DOWN
    PRESS_ARROW_LEFT = WindowEvent.PRESS_ARROW_LEFT
    PRESS_ARROW_RIGHT = WindowEvent.PRESS_ARROW_RIGHT
    PRESS_ARROW_UP = WindowEvent.PRESS_ARROW_UP
    PRESS_BUTTON_A = WindowEvent.PRESS_BUTTON_A
    PRESS_BUTTON_B = WindowEvent.PRESS_BUTTON_B
    PRESS_BUTTON_START = WindowEvent.PRESS_BUTTON_START
    
    release_actions = {
        PRESS_ARROW_DOWN: WindowEvent.RELEASE_ARROW_DOWN,
        PRESS_ARROW_LEFT: WindowEvent.RELEASE_ARROW_LEFT,
        PRESS_ARROW_RIGHT: WindowEvent.RELEASE_ARROW_RIGHT,
        PRESS_ARROW_UP: WindowEvent.RELEASE_ARROW_UP,
        PRESS_BUTTON_A: WindowEvent.RELEASE_BUTTON_A,
        PRESS_BUTTON_B: WindowEvent.RELEASE_BUTTON_B,
        PRESS_BUTTON_START: WindowEvent.RELEASE_BUTTON_START
    }
    
    

class Emulator():
    def __init__(self, gb_path: str, init_state: str, parameters: dict, headless: bool = True, max_steps: int = None, save_video: bool = None, fast_video: bool = None, session_name: str = None, instance_id: str = None):
        """_summary_
        Initializes the Pokemon Environment. 

        Args:
            gb_path (str): Path to the GameBoy ROM file.
            init_state (str): Path to the initial state file to load.
            parameters (dict): Dictionary of parameters for the environment.
            headless (bool, optional): Whether to run the environment in headless mode. Defaults to True.
            max_steps (int, optional): Maximum number of steps per episode. Defaults to None.
            save_video (bool, optional): Whether to save video of the episodes. Defaults to None.
            fast_video (bool, optional): Whether to save video in fast mode. Defaults to None.
            session_name (str, optional): Name of the session. If None, a new session name will be allocated. Defaults to None.        
        """
        assert gb_path is not None, "You must provide a path to the GameBoy ROM file."
        assert init_state is not None, "You must provide an initial state file to load."
        assert parameters is not None, "You must provide a parameters dictionary."
        assert parameters != {}, "The parameters dictionary cannot be empty."
        assert headless in [True, False], "headless must be a boolean."
        self.gb_path = gb_path
        self.init_state = init_state
        # validate init_state exists and ends with .state
        self.parameters = parameters
        if not os.path.exists(self.init_state):
            log_error(f"Initial state file {self.init_state} does not exist.", self.parameters)
        if not self.init_state.endswith(".state"):
            log_error(f"Initial state file {self.init_state} is not a .state file.", self.parameters)
        if not os.path.exists(self.gb_path):
            log_error(f"GameBoy ROM file {self.gb_path} does not exist. You must obtain a ROM through official means, and then place it in the path: {self.gb_path}", self.parameters)
        if not self.gb_path.endswith(".gb"):
            log_error(f"GameBoy ROM file {self.gb_path} is not a .gb file.", self.parameters)
        self.headless = headless
        if max_steps is None:
            max_steps = self.parameters["gameboy_max_steps"]
        if max_steps > self.parameters["gameboy_hard_max_steps"]:
            log_warn(f"max_steps {max_steps} exceeds gameboy_hard_max_steps {self.parameters['gameboy_hard_max_steps']}. Setting to hard max.", self.parameters)
            max_steps = self.parameters["gameboy_hard_max_steps"]
        self.max_steps = max_steps
        if save_video is None:
            save_video = self.parameters["gameboy_default_save_video"]
        if fast_video is None:
            fast_video = self.parameters["gameboy_default_fast_video"]
        self.save_video = save_video
        self.fast_video = fast_video
        if session_name is None:
            session_name = allocate_new_session_name(self.parameters, session_path=self.get_session_path())
        self.session_name = session_name
        self.session_path = os.path.join(self.get_session_path(), self.session_name)
        os.makedirs(self.session_path, exist_ok=True)
        if instance_id is None:
            instance_id = str(uuid.uuid4())[:8]
        self.instance_id = instance_id
        self.act_freq = parameters["gameboy_action_freq"]
        self.press_step = parameters["gameboy_press_step"]
        self.frame_stacks = parameters["gameboy_video_frame_stacks"]
        self.full_frame_writer = None
        self.model_frame_writer = None
        self.reset_count = 0
        self.step_count = 0
        self.output_shape = (72, 80, self.frame_stacks)

        head = "null" if self.headless else "SDL2"

        self.pyboy = PyBoy(
            self.gb_path,
            window=head,
        )

        #self.screen = self.pyboy.botsupport_manager().screen()

        if not self.headless:
            self.pyboy.set_emulation_speed(self.parameters["gameboy_headless_emulation_speed"])        
        
    def get_session_path(self) -> str:
        """
        Returns the path to the session directory for this environment variant.
        :return: path to the session directory
        """
        storage_dir = self.parameters["storage_dir"]
        session_path = os.path.join(storage_dir, "sessions", self.get_env_variant())
        os.makedirs(session_path, exist_ok=True)
        return session_path

    def reset(self, seed: int = None):
        """_summary_

        Args:
            seed (int, optional): Sets a random seed for the environment. Defaults to None.

        Returns:
            _type_: _description_
        """
        self.seed = seed
        # restart game, skipping to init_state 
        with open(self.init_state, "rb") as f:
            self.pyboy.load_state(f)

        self.reset_count += 1
        self.step_count = 0
        return

    def render(self, reduce_res: bool = True) -> np.ndarray:
        """_summary_
        Renders the current screen of the emulator.
        Args:
            reduce_res (bool, optional): Whether to reduce the resolution of the rendered image. Defaults to True.
        Returns:
            np.ndarray: The rendered image as a numpy array.
        """ 
        game_pixels_render = self.pyboy.screen.ndarray[:,:,0:1]  # (144, 160, 3)
        if reduce_res:
            game_pixels_render = (
                downscale_local_mean(game_pixels_render, (2,2,1))
            ).astype(np.uint8)
        return game_pixels_render
    
    def step(self, action: LowLevelActions) -> bool:
        """_summary_
        
        Takes a step in the environment by performing the given action on the emulator.
        If saving video, starts the video recording on the first step.

        Args:
            action (LowLevelActions): Lowest level action to perform on the emulator.

        Returns:
            bool: Whether the max_steps limit is reached.
        """
        if self.step_count >= self.max_steps:
            log_error("Step called after max_steps reached. Please reset the environment.", self.parameters)
            
        if self.save_video and self.step_count == 0:
            self.start_video()

        self.run_action_on_emulator(action)

        step_limit_reached = self.check_if_done()

        self.step_count += 1

        return step_limit_reached
    
    def run_action_on_emulator(self, action: LowLevelActions):
        """_summary_
        
        Performs the given action on the emulator by pressing and releasing the corresponding button.

        Args:
            action (LowLevelActions): Lowest level action to perform on the emulator.
        """
        # press button then release after some steps
        self.pyboy.send_input(action)
        # disable rendering when we don't need it
        render_screen = self.save_video or not self.headless
        press_step = self.press_step
        self.pyboy.tick(press_step, render_screen)
        self.pyboy.send_input(LowLevelActions.release_actions[action])
        self.pyboy.tick(self.act_freq - press_step - 1, render_screen)
        self.pyboy.tick(1, True)
        if self.save_video and self.fast_video:
            self.add_video_frame()

    def start_video(self):
        if self.full_frame_writer is not None:
            self.full_frame_writer.close()
        if self.model_frame_writer is not None:
            self.model_frame_writer.close()

        base_dir = os.path.join(self.session_path, "videos")
        os.makedirs(base_dir, exist_ok=True)
        full_name = os.path.join(base_dir, f"full_reset_{self.reset_count}_id{self.instance_id}.mp4")
        model_name = os.path.join(base_dir, f"model_reset_{self.reset_count}_id{self.instance_id}.mp4")
        self.full_frame_writer = media.VideoWriter(full_name, (144, 160), fps=60, input_format="gray") #TODO: check that the resolution is not specific to Pokemon Red
        self.full_frame_writer.__enter__()
        self.model_frame_writer = media.VideoWriter(model_name, self.output_shape[:2], fps=60, input_format="gray")
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

    def close(self):
        self.pyboy.stop() # TODO: check if this is the correct way to close the pyboy emulator
        if self.full_frame_writer is not None:
            self.full_frame_writer.close()
        if self.model_frame_writer is not None:
            self.model_frame_writer.close()
        # check if session directory is empty, and if so delete it
        if os.path.exists(self.session_path) and len(os.listdir(self.session_path)) == 0:
            os.rmdir(self.session_path)
    
    def save_render(self):
        render_path = os.path.join(self.session_path, "renders", f"step_{self.step_count}_id{self.instance_id}.jpeg")
        file_makedir(render_path)
        plt.imsave(render_path, self.render(reduce_res=False)[:,:, 0])
            
    def get_env_variant(self) -> str:
        """        
        Returns a string identifier for the particular environment variant being used.
        
        :return: string name identifier of the particular env e.g. PokemonRed
        """
        raise NotImplementedError