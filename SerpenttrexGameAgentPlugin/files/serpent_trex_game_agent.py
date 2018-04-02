from serpent.game_agent import GameAgent
#from serpent.game_api import GameAPI
#import time

#######################
#import serpent.cv
import random
from serpent.input_controller import KeyboardKey
import collections
import skimage.io
import skimage.transform
import skimage.measure
import skimage.util
import numpy as np
from .helpers.helper import expand_bounding_box
from .helpers.terminal_printer import TerminalPrinter
from serpent.game_frame import GameFrame
#from serpent.frame_grabber import FrameGrabber
#from serpent.config import config
#######################
###########################################################################
from serpent.game_agent import GameAgent

from serpent.game_frame import GameFrame
#from serpent.frame_grabber import FrameGrabber

from serpent.input_controller import KeyboardKey

#from serpent.config import config

from datetime import datetime

import skimage.io
import skimage.transform
import skimage.measure
import skimage.util

import numpy as np

import random
#import time
import collections
#import subprocess
#import shlex
import os

import serpent.ocr
import serpent.cv
import serpent.utilities

from .helpers.helper import expand_bounding_box
from .helpers.terminal_printer import TerminalPrinter

#from .helpers.ppo import SerpentPPO
###########################################################################
from serpent.machine_learning.reinforcement_learning.keyboard_mouse_action_space import KeyboardMouseActionSpace
import gc
from serpent.machine_learning.reinforcement_learning.ddqn import DDQN
from colorama import init, Fore, Back, Style
import serpent.cv
import time
import pytesseract
###########################################################################

class SerpenttrexGameAgent(GameAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.frame_handlers["PLAY"] = self.handle_play

        self.frame_handler_setups["PLAY"] = self.setup_play

        self.analytics_client = None

        self.game_state = None
        self._reset_game_state()

        init()

    def setup_play(self):
        #print(f"### setup_play")
        self.performed_inputs = collections.deque(list(), maxlen=8)
        self.game_inputs = {
            "JUMP": [KeyboardKey.KEY_SPACE],
            "DUCK": [KeyboardKey.KEY_DOWN]
            # "NOOP": []
        }

        self.key_mapping = {
            KeyboardKey.KEY_SPACE: "JUMP",
            KeyboardKey.KEY_DOWN: "DUCK"
        }

        # Game Specific Inputs
        direction_action_space = KeyboardMouseActionSpace(
            direction_keys=[None, "JUMP", "DUCK"])

        direction_model_file_path = "datasets/trex_direction_dqn_0_1.hf".replace("/", os.sep)

        self.dqn_direction = DDQN(
            model_file_path=direction_model_file_path if os.path.isfile(direction_model_file_path) else None,
            input_shape=(400, 400, 4),
            input_mapping=self.game_inputs,
            action_space=direction_action_space,
            replay_memory_size=40000,
            max_steps=3000000,
            observe_steps=5000,
            batch_size=32,
            model_learning_rate=1e-4,
            initial_epsilon=1.0,
            final_epsilon=0.1,
            override_epsilon=False
        )



    def setup_random(self):
        #print(f"### setup_random")
        self.performed_inputs = collections.deque(list(), maxlen=8)
        #pass

    def handle_play(self, game_frame):
        #print(f"### handle_play")
        gc.disable()

        if self.dqn_direction.first_run:
            print(f"### First run, tap space to get us going...")
            self.input_controller.tap_key(KeyboardKey.KEY_SPACE)

            self.dqn_direction.first_run = False

            return None

        # need to try to pull score
        curr_score = self._get_score(game_frame)
        high_score = self._get_high_score(game_frame)
        print(f"#### curr_score: {curr_score}")
        print(f"#### high_score: {high_score}")

        
        my_image = self.game.api._capture_game_over_image()
        
        # start visual debugger in another window:
        # run: serpent visual_debugger 0
        # self.visual_debugger.store_image_data(
        #     my_image,
        #     my_image.shape,
        #     bucket="0"
        # )
        
        game_input_key = random.choice(list(self.game_inputs.keys()))

        self.performed_inputs.appendleft(game_input_key)

        self.input_controller.handle_keys(self.game_inputs[game_input_key])

        self.is_alive([None, None, game_frame, None])


    def handle_random(self, game_frame):
        #print(f"## handle_random...")
        game_input_key = random.choice(list(self.game_inputs.keys()))

        self.performed_inputs.appendleft(game_input_key)

        self.input_controller.handle_keys(self.game_inputs[game_input_key])

    def is_alive(self,frames,**kwargs):
        #print(f"### is_alive")
        bw_frame = self.get_clean_frame(frames)
        if not self.game.api.is_alive(GameFrame(frames[-2].frame), self.sprite_identifier):
            print(f"### Died...")
            self.input_controller.tap_key(KeyboardKey.KEY_SPACE)
            return 0

        else:
            # print(f"### Still Alive...")
            pass

    def get_clean_frame(self,frames):
        #print(f"### get_clean_frame")
        bw_frame = skimage.util.img_as_ubyte(np.all(frames[-2].frame > 240, axis=-1))

        label_frame = skimage.measure.label(bw_frame)
        regions = skimage.measure.regionprops(label_frame)

        clean_frame = np.zeros(bw_frame.shape, dtype="uint8")

        bounding_boxes = list()

        for region in regions:
            if region.area <= 10 or region.area > 1000:
                continue

            y0, y1, x0, x1 = expand_bounding_box(region.bbox, bw_frame.shape, 5, 5)
            aspect_ratio = (x1 - x0) / (y1 - y0)

            if aspect_ratio < 0.4 or aspect_ratio > 1.0:
                continue

            bounding_boxes.append([y0, y1, x0, x1])

        for b in bounding_boxes:
            for bb in bounding_boxes:
                if b == bb:
                    continue

                if b[0] in range(bb[0], bb[1] + 1) or b[1] in range(bb[0], bb[1] + 1):
                    if b[2] in range(bb[2], bb[3] + 1) or b[3] in range(bb[2], bb[3] + 1):
                        clean_frame[b[0]:b[1], b[2]:b[3]] = 255
                        break

        return clean_frame


    def _reset_game_state(self):
        self.game_state = {
            "health": collections.deque(np.full((8,), 3), maxlen=8),
            "score": collections.deque(np.full((8,), 0), maxlen=8),
            "run_reward_direction": 0,
            "run_reward_action": 0,
            "current_run": 1,
            "current_run_steps": 0,
            "current_run_health": 3,
            "current_run_score": 0,
            "run_predicted_actions": 0,
            "last_run_duration": 0,
            "record_time_alive": dict(),
            "random_time_alive": None,
            "random_time_alives": list(),
            "run_timestamp": datetime.utcnow(),
        }

    def _get_score(self,game_frame):
        score_area_frame = serpent.cv.extract_region_from_image(game_frame.frame, self.game.screen_regions["SCORE_AREA"])
        score_grayscale = np.array(skimage.color.rgb2gray(score_area_frame) * 255, dtype="uint8")

        score = pytesseract.image_to_string(score_grayscale, config='-psm 6')
        return score

    def _get_high_score(self,game_frame):
        score_area_frame = serpent.cv.extract_region_from_image(game_frame.frame, self.game.screen_regions["HIGH_SCORE_AREA"])
        score_grayscale = np.array(skimage.color.rgb2gray(score_area_frame) * 255, dtype="uint8")

        score = pytesseract.image_to_string(score_grayscale, config='-psm 6')
        return score
