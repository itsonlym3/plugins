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
from serpent.frame_grabber import FrameGrabber
import skimage.color

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
        # print(f"#### curr_score: {curr_score}")
        # print(f"#### high_score: {high_score}")

        
        self.game_state["score"].appendleft(curr_score)


        if self.dqn_direction.frame_stack is None:
            pipeline_game_frame = FrameGrabber.get_frames(
                [0],
                frame_shape=(self.game.frame_height, self.game.frame_width),
                frame_type="PIPELINE"
            ).frames[0]

            self.dqn_direction.build_frame_stack(pipeline_game_frame.frame)
            #self.dqn_action.frame_stack = self.dqn_direction.frame_stack
        else:
            game_frame_buffer = FrameGrabber.get_frames(
                [0, 4, 8, 12],
                frame_shape=(self.game.frame_height, self.game.frame_width),
                frame_type="PIPELINE"
            )

            if self.dqn_direction.mode == "TRAIN":
                reward_direction, reward_action = self._calculate_reward()

                self.game_state["run_reward_direction"] += reward_direction
                #self.game_state["run_reward_action"] += reward_action

                self.dqn_direction.append_to_replay_memory(
                    game_frame_buffer,
                    reward_direction,
                    terminal=self.game_state["score"] == 0
                )

                # self.dqn_action.append_to_replay_memory(
                #     game_frame_buffer,
                #     reward_action,
                #     terminal=self.game_state["health"] == 0
                # )

                # Every 2000 steps, save latest weights to disk
                #if self.dqn_direction.current_step % 2000 == 0:
                if self.dqn_direction.current_step % 200 == 0:
                    print("### Writing Data...")
                    self.dqn_direction.save_model_weights(
                        file_path_prefix=f"datasets/trex_direction"
                    )

                    # self.dqn_action.save_model_weights(
                    #     file_path_prefix=f"datasets/spaceinvaders_action"
                    # )

                # Every 20000 steps, save weights checkpoint to disk
                #if self.dqn_direction.current_step % 20000 == 0:
                if self.dqn_direction.current_step % 2000 == 0:
                    self.dqn_direction.save_model_weights(
                        file_path_prefix=f"datasets/trex_direction",
                        is_checkpoint=True
                    )

                    # self.dqn_action.save_model_weights(
                    #     file_path_prefix=f"datasets/spaceinvaders_action",
                    #     is_checkpoint=True
                    # )
            elif self.dqn_direction.mode == "RUN":
                self.dqn_direction.update_frame_stack(game_frame_buffer)
                # self.dqn_action.update_frame_stack(game_frame_buffer)

            run_time = datetime.now() - self.started_at

            serpent.utilities.clear_terminal()

            print("\033[31m" + f"SESSION RUN TIME: {run_time.days} days, {run_time.seconds // 3600} hours, {(run_time.seconds // 60) % 60} minutes, {run_time.seconds % 60} seconds" + "\033[37m")
            print("GAME: T-Rex Rush   PLATFORM: Windows/Python   AGENT: DDQN + PER")
            print("BE WARNED!!!!  I HAVE NOOOOO IDEA WHAT I AM DOING.")
            print("")

            print("\033[32m" + "DIRECTION NEURAL NETWORK INFO:\n" + "\033[37m")
            self.dqn_direction.output_step_data()

            # print("")
            # print("\033[32m" + "ACTION NEURAL NETWORK INFO:\n" + "\033[37m")
            # self.dqn_action.output_step_data()

            print("")
            print(f"CURRENT RUN: {self.game_state['current_run']}")
            print(f"CURRENT RUN REWARD: {round(self.game_state['run_reward_direction'])}")
            # print(f"CURRENT RUN REWARD: {round(self.game_state['run_reward_direction'] + self.game_state['run_reward_action'], 2)}")
            # print(f"CURRENT RUN PREDICTED ACTIONS: {self.game_state['run_predicted_actions']}")
            # print(f"CURRENT HEALTH: {self.game_state['health'][0]}")
            print(f"CURRENT SCORE: {self.game_state['score'][0]}")
            # print(f"CURRENT CREDITS: {self.game_state['credits'][0]}")
            print("")
            print(f"LAST RUN DURATION: {self.game_state['last_run_duration']} seconds")

            print("")
            print(f"RECORD TIME ALIVE: {self.game_state['record_time_alive'].get('value')} seconds (Run {self.game_state['record_time_alive'].get('run')}, {'Predicted' if self.game_state['record_time_alive'].get('predicted') else 'Training'})")
            print("")

            print(f"RANDOM AVERAGE TIME ALIVE: {self.game_state['random_time_alive']} seconds")

            #tjb this is where we need to match the restart image (sprite_trex_game_over_0.png).
            #if self.game_state["health"][2] <= 0:
            if self.is_alive([None, None, game_frame, None]) == 0:
                #serpent.utilities.clear_terminal()
                #print("ENTERING THE HEALTH <= 0 PART")
                print("WE DEADs")
                timestamp = datetime.utcnow()

                gc.enable()
                gc.collect()
                gc.disable()

                timestamp_delta = timestamp - self.game_state["run_timestamp"]
                self.game_state["last_run_duration"] = timestamp_delta.seconds

                if self.dqn_direction.mode in ["TRAIN", "RUN"]:
                    # Check for Records
                    if self.game_state["last_run_duration"] > self.game_state["record_time_alive"].get("value", 0):
                        self.game_state["record_time_alive"] = {
                            "value": self.game_state["last_run_duration"],
                            "run": self.game_state["current_run"],
                            "predicted": self.dqn_direction.mode == "RUN"
                        }
                else:
                    self.game_state["random_time_alives"].append(self.game_state["last_run_duration"])
                    self.game_state["random_time_alive"] = np.mean(self.game_state["random_time_alives"])

                self.game_state["current_run_steps"] = 0

                self.input_controller.handle_keys([])

                if self.dqn_direction.mode == "TRAIN":
                    for i in range(16):
                        serpent.utilities.clear_terminal()
                        print("\033[31m" + f"SESSION RUN TIME: {run_time.days} days, {run_time.seconds // 3600} hours, {(run_time.seconds // 60) % 60} minutes, {run_time.seconds % 60} seconds" + "\033[37m")
                        print("\033[32m" + "GAME: Space Invaders   PLATFORM: Steam   AGENT: DDQN + PER" + "\033[37m")
                        print("")
                        print("TRAINING ON MINI-BATCHES:" + "\033[32m" +  f"{i + 1}/16" + "\033[37m")
                        print(f"NEXT RUN: {self.game_state['current_run'] + 1} {'- AI RUN' if (self.game_state['current_run'] + 1) % 20 == 0 else ''}")

                        self.dqn_direction.train_on_mini_batch()
                        #self.dqn_action.train_on_mini_batch()

                self.game_state["run_timestamp"] = datetime.utcnow()
                self.game_state["current_run"] += 1
                self.game_state["run_reward_direction"] = 0
                self.game_state["run_reward_action"] = 0
                self.game_state["run_predicted_actions"] = 0
                #self.game_state["health"] = collections.deque(np.full((8,), 3), maxlen=8)
                self.game_state["score"] = collections.deque(np.full((8,), 0), maxlen=8)

                if self.dqn_direction.mode in ["TRAIN", "RUN"]:
                    if self.game_state["current_run"] > 0 and self.game_state["current_run"] % 100 == 0:
                        if self.dqn_direction.type == "DDQN":
                            self.dqn_direction.update_target_model()
                            self.dqn_action.update_target_model()

                    if self.game_state["current_run"] > 0 and self.game_state["current_run"] % 20 == 0:
                        self.dqn_direction.enter_run_mode()
                        #self.dqn_action.enter_run_mode()
                    else:
                        self.dqn_direction.enter_train_mode()
                        #self.dqn_action.enter_train_mode()

                time.sleep(1)
                self.input_controller.tap_key(KeyboardKey.KEY_SPACE)
                # self.input_controller.tap_key(KeyboardKey.KEY_N)
                # time.sleep(1)
                # self.input_controller.tap_key(KeyboardKey.KEY_ENTER)
                # time.sleep(6)

                return None



















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
            #print(f"### Died...")
            self.input_controller.tap_key(KeyboardKey.KEY_SPACE)
            return 0

        else:
            # print(f"### Still Alive...")
            return 1
            #pass

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
        # print(f"### _get_score: {score}")
        return score

    def _get_high_score(self,game_frame):
        score_area_frame = serpent.cv.extract_region_from_image(game_frame.frame, self.game.screen_regions["HIGH_SCORE_AREA"])
        score_grayscale = np.array(skimage.color.rgb2gray(score_area_frame) * 255, dtype="uint8")

        score = pytesseract.image_to_string(score_grayscale, config='-psm 6')
        # print(f"### _get_high_score: {score}")
        return score

    def _calculate_reward(self):
        reward = 0

        # reward += (-1.0 if self.game_state["credits"][0] < self.game_state["credits"][1] else 0.1)
        #reward += (-0.5 if self.game_state["health"][0] < self.game_state["health"][1] else 0.05)
        reward += (0.75 if (int(self.game_state["score"][0]) - int(self.game_state["score"][1])) >= 10 else -0.075)


        #return reward, reward
        return reward
