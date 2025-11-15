from enum import Enum
from abc import ABC, abstractmethod
from typing import Type


import os
from time import perf_counter
import sys
import shutil
import uuid
from poke_env.utils import load_parameters, log_error, log_warn, file_makedir, log_info, is_none_str, nested_dict_to_str


import cv2
from pyboy import PyBoy
from pyboy.utils import WindowEvent
from matplotlib import pyplot as plt
from skimage.transform import downscale_local_mean
import numpy as np



class LowLevelActions(Enum):
    """
    Enum for low-level actions that can be performed on the GameBoy emulator and their associated release actions.
    """
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
        PRESS_BUTTON_START: WindowEvent.RELEASE_BUTTON_START}


class GameStateParser(ABC):
    """
    Abstract base class for parsing game state variables from the GameBoy emulator.
    """
    def __init__(self, pyboy, parameters):
        self._pyboy = pyboy
        self._parameters = parameters
        self.parsed_variables = {"done": False}
        """Dictionary to hold all parsed game state variables."""

    def clear(self):
        """
        Resets the parsed variables dictionary. Typically called at the start of a new episode.
        """
        self.parsed_variables = {"done": False}

    @abstractmethod
    def parse_step(self):
        """
        Parses the game state at the current step. Saves any relevant variables to `self.parsed_variables`
        """
        raise NotImplementedError

    @abstractmethod
    def parse_all(self):
        """
        Parses the game state. Use this function for additional variables that you may not need at every step because they are expensive to compute.
        """
        raise NotImplementedError

    @abstractmethod
    def __repr__(self) -> str:
        """
        Name of the parser for logging purposes.
        :return: string name of the parser
        """
        raise NotImplementedError
    
    def __str__(self) -> str:
        start = f"***\tGameStateParser({self.__repr__()})\t***"
        body = nested_dict_to_str(self.parsed_variables, indent=1)
        return f"{start}\n{body}"
    


class Emulator(ABC):
    def __init__(self, gb_path: str, game_state_parser_class: Type[GameStateParser], init_state: str, parameters: dict, headless: bool = True, max_steps: int = None, save_video: bool = None, session_name: str = None, instance_id: str = None):
        """
        Start the GameBoy emulator with the given ROM file and initial state.

        Args:
            gb_path (str): Path to the GameBoy ROM file.
            game_state_parser_class (Type[GameStateParser]): A class that inherits from GameStateParser to parse game state variables.
            init_state (str): Path to the initial state file to load.
            parameters (dict): Dictionary of parameters for the environment.
            headless (bool, optional): Whether to run the environment in headless mode. Defaults to True.
            max_steps (int, optional): Maximum number of steps per episode. Defaults to None.
            save_video (bool, optional): Whether to save video of the episodes. Defaults to None.            
            session_name (str, optional): Name of the session. If None, a new session name will be allocated. Defaults to None.        
        """
        assert gb_path is not None, "You must provide a path to the GameBoy ROM file."
        assert isinstance(game_state_parser_class, type) and issubclass(game_state_parser_class, GameStateParser), "You must provide a valid GameStateParser subclass."
        assert init_state is not None, "You must provide an initial state file to load."
        assert parameters is not None, "You must provide a parameters dictionary."
        assert parameters != {}, "The parameters dictionary cannot be empty."
        assert headless in [True, False], "headless must be a boolean."
        self._gb_path = gb_path
        self._set_init_state(init_state)
        # validate init_state exists and ends with .state
        self._parameters = parameters
        if not os.path.exists(self._gb_path):
            log_error(f"GameBoy ROM file {self._gb_path} does not exist. You must obtain a ROM through official means, and then place it in the path: {self._gb_path}", self._parameters)
        if not self._gb_path.endswith(".gb"):
            log_error(f"GameBoy ROM file {self._gb_path} is not a .gb file.", self._parameters)
        self.headless = headless
        """ Whether to run the environment in headless mode."""
        if max_steps is None:
            max_steps = self._parameters["gameboy_max_steps"]
        if max_steps > self._parameters["gameboy_hard_max_steps"]:
            log_warn(f"max_steps {max_steps} exceeds gameboy_hard_max_steps {self._parameters['gameboy_hard_max_steps']}. Setting to hard max.", self._parameters)
            max_steps = self._parameters["gameboy_hard_max_steps"]
        self.max_steps = max_steps
        """ Maximum number of steps per episode. """
        if save_video is None:
            save_video = self._parameters["gameboy_default_save_video"]
        self.save_video = save_video
        """ Whether to save video of the episodes. """
        if session_name is None:
            session_name = self._allocate_new_session_name()
        self.session_name = session_name
        """ Name of the session. Decides the directory where artifacts are saved. """
        self.session_path = os.path.join(self.get_session_path(), self.session_name)
        """ Path to the session directory. This is where all artifacts for this session are saved. """
        os.makedirs(self.session_path, exist_ok=True)
        if instance_id is None:
            instance_id = str(uuid.uuid4())[:8]
        self.instance_id = instance_id
        """ Unique identifier for this environment instance. Useful for distinguishing multiple environments running in parallel. """
        self.act_freq = parameters["gameboy_action_freq"]
        """ Number of emulator ticks per action. Defaults to value specified in config files. """
        self.press_step = parameters["gameboy_press_step"]
        """ Number of emulator ticks to hold down a button press. Defaults to value specified in config files. """
        self.frame_stacks = parameters["gameboy_video_frame_stacks"]
        """ TODO: IDK YET """
        self.render_headless = parameters["gameboy_headless_render"]
        """ Whether to render the emulator screen even in headless mode. This must be true for methods that rely on image observations (e.g. VLMs) to access the screen. Defaults to value specified in config files. """
        self._full_frame_writer = None
        self._model_frame_writer = None
        self.reset_count = 0
        """ Number of times the environment has been reset. """
        self.step_count = 0
        """ Number of steps taken in the current episode. """
        self._reduce_video_resolution = parameters["gameboy_reduce_video_resolution"]
        pokemon_frame_size = (160, 144) # TODO: confirm this is universal if you want other GB games. 
        if self._reduce_video_resolution:
            self.output_shape = (pokemon_frame_size[0]//2, pokemon_frame_size[1]//2, self.frame_stacks)
        else:
            self.output_shape = (pokemon_frame_size[0], pokemon_frame_size[1], self.frame_stacks)
            """ Shape of the output observations. This is the resolution of the rendered screen. """

        head = "null" if self.headless else "SDL2"

        self._pyboy = PyBoy(
            self._gb_path,
            window=head,
        )
        self.game_state_parser = game_state_parser_class(self._pyboy, self._parameters)
        """ Instance of the GameStateParser to parse game state variables. """

        #self.screen = self.pyboy.botsupport_manager().screen()

        if not self.headless:
            if not is_none_str(self._parameters["gameboy_headed_emulation_speed"]):
                self._pyboy.set_emulation_speed(int(self._parameters["gameboy_headed_emulation_speed"]))        
            
    def _allocate_new_session_name(self) -> str:
        """
        Allocates a new session name based on existing sessions in the session directory.
        :return: new session name
        """
        storage_dir = self._parameters["storage_dir"]
        session_path = os.path.join(storage_dir, "sessions", self.get_env_variant())
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

    def clear_unnamed_sessions(self):
        """
        Clears all unnamed (integer based) sessions from the session directory.
        """
        storage_dir = self._parameters["storage_dir"]
        session_path = os.path.join(storage_dir, "sessions", self.get_env_variant())
        if not os.path.exists(session_path):
            return
        existing_sessions = os.listdir(session_path)
        saved_sessions = [self.session_name]
        for session in existing_sessions:
            if session.startswith("session_"):
                session_id = session.split("_")[-1]
                if session_id.isdigit():
                    full_path = os.path.join(session_path, session)
                    # delete the session directory and all its contents
                    import shutil
                    shutil.rmtree(full_path)
                    log_info(f"Deleted unnamed session {full_path}", self._parameters)
                    continue
            saved_sessions.append(session)
        log_info(f"Kept sessions: {saved_sessions}", self._parameters)
                
    def get_session_path(self) -> str:
        """
        Returns the path to the session directory for this environment variant.
        :return: path to the session directory
        """
        storage_dir = self._parameters["storage_dir"]
        session_path = os.path.join(storage_dir, "sessions", self.get_env_variant())
        os.makedirs(session_path, exist_ok=True)
        return session_path

    def set_init_state(self, init_state: str):
        """Sets a new initial state file for the environment. and resets the environment.

        Args:
            init_state (str): Path to the new initial state file.
        """
        self._set_init_state(init_state)
        self.reset()
    
    def _set_init_state(self, init_state: str):
        """
        Sets a new initial state file for the environment to eventually load. 
        Does not reset the environment. 

        Args:
            init_state (str): Path to the new initial state file.
        """
        if init_state is None:
            return
        if not os.path.exists(init_state):
            log_error(f"New initial state file {init_state} does not exist.", self._parameters)
        if not init_state.endswith(".state"):
            log_error(f"New initial state file {init_state} is not a .state file.", self._parameters)
        self.init_state = init_state

    def reset(self, new_init_state: str = None, seed: int = None):
        """
        Resets the environment to the initial state. Optionally loads a new initial state file and sets a random seed.

        Args:
            new_init_state (str, optional): Path to a new initial state file to load. Defaults to None.
            seed (int, optional): Sets a random seed for the environment. Defaults to None. TODO: implement seeding.

        Returns:
            _type_: _description_
        """
        self.seed = seed # TODO: implement seeding
        # validate the new_init_state if provided
        if new_init_state is not None:
            self._set_init_state(new_init_state)
        # restart game, skipping to init_state 
        with open(self.init_state, "rb") as f:
            self._pyboy.load_state(f)

        self.reset_count += 1
        self.step_count = 0
        self.close_video()
        return

    def get_current_frame(self, reduce_res: bool = None) -> np.ndarray:
        """
        Renders the currently rendered screen of the emulator and returns it as a numpy array.
        Args:
            reduce_res (bool, optional): Whether to reduce the resolution of the rendered image. Defaults to gameboy_reduce_video_resolution.
        Returns:
            np.ndarray: The rendered image as a numpy array.
        """ 
        game_pixels_render = self._pyboy.screen.ndarray[:,:,0:1]  # (144, 160, 3)
        if reduce_res is None:
            reduce_res = self._reduce_video_resolution
        if reduce_res:
            game_pixels_render = (
                downscale_local_mean(game_pixels_render, (2,2,1))
            ).astype(np.uint8)
        return game_pixels_render
    
    def step(self, action: LowLevelActions = None) -> bool:
        """ 
        
        Takes a step in the environment by performing the given action on the emulator.
        If saving video, starts the video recording on the first step.

        Args:
            action (LowLevelActions, optional): Lowest level action to perform on the emulator.

        Returns:
            bool: Whether the max_steps limit is reached.
        """
        if action is not None:
            if action not in LowLevelActions:
                log_error(f"Invalid action {action}. Must be one of {list(LowLevelActions)} or None", self._parameters)
        if self.step_count >= self.max_steps:
            log_error("Step called after max_steps reached. Please reset the environment.", self._parameters)
            
        if self.save_video and self.step_count == 0:
            self.start_video()
        
        if self.save_video or self.video_running: # TODO: Consider alternative ways of handling this
            self.add_video_frame()

        self.run_action_on_emulator(action)
        self.game_state_parser.parse_step()

        if self.check_if_done():
            self.game_state_parser.parsed_variables["done"] = True
        print(self.game_state_parser)

        self.step_count += 1

        return self.game_state_parser

    def get_game_state(self) -> GameStateParser:
        """
        Returns the current game state parser instance.

        Returns:
            GameStateParser: The current game state parser.
        """
        return self.game_state_parser

    def run_action_on_emulator(self, action: LowLevelActions = None, profile: bool = False, render: bool = None):
        """ 
        
        Performs the given action on the emulator by pressing and releasing the corresponding button.

        Args:
            action (LowLevelActions): Lowest level action to perform on the emulator.
        """
        # press button then release after some steps
        #log_info(f"Running action: {action}", self.parameters)        
        if action is not None:
            if profile:
                start_time = perf_counter()
            self._pyboy.send_input(action.value)
            if profile:
                end_time = perf_counter()
                # Action LowLevelActions.PRESS_ARROW_LEFT took 0.01 ms or 0.00 s

            # disable rendering when we don't need it
            if render is not None:
                render_screen = render
            else:
                render_screen = self.save_video or not self.headless or self.render_headless
            press_step = self.press_step
            self._pyboy.tick(press_step, render_screen)
            if profile:
                start_time = perf_counter()
            self._pyboy.send_input(LowLevelActions.release_actions.value[action.value])
            if profile:
                mid_time = perf_counter()
            self._pyboy.tick(self.act_freq - press_step - 1, render_screen)
            if profile:
                end_time = perf_counter()
            # Releasing action LowLevelActions.PRESS_ARROW_LEFT took 0.00 ms, followed by 16.13 ms for remaining ticks
            self._pyboy.tick(1, True)
        else:
            log_warn("No action provided to run_action_on_emulator. Skipping action. You probably should only ever use this in debugging mode. Idk maybe wait is a command.", self._parameters)
            self._pyboy.tick(self.act_freq, True)
    
    def get_free_video_id(self) -> str:
        """
        Returns a new unique video ID for saving video files.

        Returns:
            str: A new unique video ID.
        """
        base_dir = os.path.join(self.session_path, "videos")
        videos = os.listdir(base_dir) if os.path.exists(base_dir) else []
        # all will be something.mp4, if its int.mp4, get the int
        video_ints = []
        for video in videos:
            if video.endswith(".mp4"):
                video_name = video[:-4]
                if video_name.isdigit():
                    video_ints.append(int(video_name))
        if len(video_ints) == 0:
            return "0.mp4"
        return str(max(video_ints) + 1)+".mp4"
        
    def start_video(self, video_id: str = None):
        """ 
        Starts recording video of the emulator's screen.
        Args:
            video_id (str, optional): Name of the video file to save. If None, a new unique name will be generated. Defaults to None.
        """
        if video_id is not None:
            if not isinstance(video_id, str):
                log_error("video_id must be a string (not digits) if provided.", self._parameters)
            if not video_id.endswith(".mp4"):
                log_error("video_id must end with .mp4 if provided.", self._parameters)
            if os.path.exists(os.path.join(self.session_path, "videos", video_id)):
                log_warn(f"video_id {video_id} already exists. Overwriting...", self._parameters)                       
        else:
            video_id = self.get_free_video_id()
        base_dir = os.path.join(self.session_path, "videos")
        os.makedirs(base_dir, exist_ok=True)
        model_name = os.path.join(base_dir, f"{video_id}")
        self.close_video()
        self.frame_writer = cv2.VideoWriter(model_name, cv2.VideoWriter_fourcc(*"mp4v"), 60, (self.output_shape[0], self.output_shape[1]), isColor=False)
        self.video_running = True

    def add_video_frame(self):
        """
        Adds the current frame of the emulator to the video being recorded.
        """
        current_frame = self.get_current_frame(reduce_res=self._reduce_video_resolution)[:, :, 0]
        # frame_size = (current_frame.shape[1], current_frame.shape[0]) # Width, Height, should be equal to self.output_shape[:2]
        # Create VideoWriter object
        frame = current_frame
        self.frame_writer.write(frame)
                    
    def check_if_done(self):
        """
        Checks if the max_steps limit has been reached.
        """
        done = self.step_count >= self.max_steps - 1
        return done

    def close_video(self):
        """
        Closes the video writer and stops recording video.
        """
        if self._full_frame_writer is not None:
            self._full_frame_writer.release()
            self._full_frame_writer = None
        if self._model_frame_writer is not None:
            self._model_frame_writer.release()
            self._model_frame_writer = None
        self.video_running = False
    
    def close(self):
        """
        Closes the emulator and any associated resources.
        If the session directory is empty after closing, it will be deleted.
        """
        self._pyboy.stop(save=False)
        self.close_video()
        # check if session directory is empty, and if so delete it
        if os.path.exists(self.session_path) and len(os.listdir(self.session_path)) == 0:
            os.rmdir(self.session_path)
    
    def human_play(self, max_steps: int = None):
        """ 
        Allows a human to play the emulator using keyboard inputs.
        Args:
            max_steps (int, optional): Maximum number of steps to play. Defaults to gameboy_hard_max_steps in configs.
        """
        if max_steps is None:
            max_steps = self._parameters["gameboy_hard_max_steps"]
        log_info("Starting human play mode. Use arrow keys and A/B/Start buttons to play. Close the window to exit.", self._parameters)
        if self.headless:
            log_error("Human play mode requires headless=False. Change the initialization", self._parameters)    
        self.reset()
        self.game_state_parser.parse_step()
        starting_state = str(self.game_state_parser)
        while True:
            self._pyboy.tick(1, True)
            self.game_state_parser.parse_step()
            new_state = str(self.game_state_parser)
            if new_state != starting_state:
                log_info(f"Current game state:\n{new_state}", self._parameters)
                starting_state = new_state
            if self.step_count >= max_steps:
                break

        self.close()
        # wait for pyboy
        #sleep(1) TODO: See how to close properly in this setting. 
        
    def _human_step_play(self, max_steps: int = None, init_state: str = None):
        """ 
        Allows a human to play the emulator using keyboard inputs. This routes the code through the step function. Only for debugging.         
        Args:
            max_steps (int, optional): Maximum number of steps to play. Defaults to gameboy_hard_max_steps in configs.
        """
        if self.headless:
            log_error("Human play mode requires headless=False. Change the initialization", self._parameters)    
        if max_steps is None:
            max_steps = self._parameters["gameboy_hard_max_steps"]
        self._set_init_state(init_state)
        character_to_action = {
            "a": LowLevelActions.PRESS_BUTTON_A,
            "b": LowLevelActions.PRESS_BUTTON_B,
            "s": LowLevelActions.PRESS_BUTTON_START,
            "u": LowLevelActions.PRESS_ARROW_UP,
            "d": LowLevelActions.PRESS_ARROW_DOWN,
            "l": LowLevelActions.PRESS_ARROW_LEFT,
            "r": LowLevelActions.PRESS_ARROW_RIGHT,            
            "": None,
        }
        msg = "Starting human play mode. Use keyboard inputs: a,b,s,u,d,l,r to play. Type v to start recording, c to end recording, i1 to save state and i2 to identify the addresses that have changed, and e to exit."
        log_info(msg, self._parameters)
        log_info(f"Character to action mapping: \n{character_to_action}", self._parameters)
        self.reset()
        exited = False
        while not exited:
            user_input = input(msg + "\nYour input: ")
            user_input = user_input.lower().strip()
            if user_input == "e":
                exited = True
                log_info("Exiting human play mode.", self._parameters)
                break
            if user_input == "v":
                self.start_video()
                continue
            if user_input == "c":
                self.close_video()
                log_info("Stopped recording video.", self._parameters)                
                continue
            if user_input == "i":
                    breakpoint()
                    continue
            if user_input not in character_to_action:
                log_warn(f"Invalid input {user_input}. Valid inputs are: {list(character_to_action.keys())} or e to exit.", self._parameters)
                continue
            action = character_to_action[user_input]
            state = self.step(action)
            if state.parsed_variables["done"]:
                log_info("Max steps reached. Exiting human play mode.", self._parameters)
                break
        
    def save_render(self):
        """
        Saves the current rendered screen of the emulator as a JPEG image in the renders directory.
        """
        render_path = os.path.join(self.session_path, "renders", f"step_{self.step_count}_id{self.instance_id}.jpeg")
        file_makedir(render_path)
        plt.imsave(render_path, self.render(reduce_res=False)[:,:, 0])

    def save_state(self, state_path: str):
        """
        Saves the current state of the emulator to a .state file.
        Args:
            state_path (str): Path to save the .state file.

        """
        if not state_path.endswith(".state"):
            state_path = state_path + ".state"
        with open(state_path, "wb") as f:
            self._pyboy.save_state(f)
        log_info(f"Saved state to {state_path}", self._parameters)

    def _sav_to_state(self, save_path: str):
        """
        Loads a .sav file into the emulator and saves the corresponding .state file. 
        Use this if you want to manually create .sav files and convert them to .state files for use as initial states.

        Args:
            save_path (str): Path to save the .state file
        Expects there to be a .sav file with the same path+name as the self.gb_path (but with .sav extension).
        """
        log_info("Trying to find .sav file and convert to .state file. This is a breaking operation, so the program will terminate after its completion.", self._parameters)
        expected_sav = self._gb_path.replace(".gb", ".sav")
        if not os.path.exists(expected_sav):
            log_error(f"Expected .sav file at {expected_sav} to convert to .state file, but it does not exist.", self._parameters)
        if save_path is None or save_path == "":
            log_error("You must provide a save_path to save the .state file.", self._parameters)
        file_makedir(save_path)
        # copy the .sav file to a .gb.ram file
        shutil.copyfile(expected_sav, expected_sav.replace(".sav", ".gb.ram"))
        self.close()
        self._pyboy = PyBoy(
            self._gb_path,
            window="null",
        )
        self._pyboy.set_emulation_speed(0)
        self._pyboy.tick(10000, False) # get to opening menu
        self.run_action_on_emulator(LowLevelActions.PRESS_BUTTON_A, render=True) # press A to get past opening menu
        self._pyboy.tick(1000, False) # wait for load
        self.run_action_on_emulator(LowLevelActions.PRESS_BUTTON_A, render=True) # press A to load game
        self._pyboy.tick(1000, False) # wait for file select
        self.run_action_on_emulator(LowLevelActions.PRESS_BUTTON_A, render=True) # press A to confirm load
        self._pyboy.tick(5000, False) # wait for game to load
        self.save_state(save_path)
        self._pyboy.stop(save=False)
        log_info("Exiting now to avoid issues.", self._parameters)
        sys.exit(0)
            
    @abstractmethod
    def get_env_variant(self) -> str:
        """        
        Returns a string identifier for the particular environment variant being used.
        
        :return: string name identifier of the particular env e.g. PokemonRed
        """
        raise NotImplementedError
    

def bytes_to_padded_hex_string(integer_value):
    """
    Converts a bytes object into a padded, '0x'-prefixed hexadecimal string.
    """
    # 1. Convert the bytes object back into an integer
    # Assumes big-endian order for the example '0x00a' -> 10
    # 2. Format the integer into a string with padding and the '0x' prefix
    # The 'x' specifier for hex, '#' adds '0x', '04' pads to 4 hex characters total
    # (not including the '0x' prefix for simple formatters like this, but managing width)
    
    # A robust approach to match your exact output '0x00a':
    # You generally want enough width for your bytes. b'\n' is 1 byte, 2 hex chars.

    return f'0x{integer_value:04x}' # {0:04x} pads to 4 digits specifically