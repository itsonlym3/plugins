from serpent.game_api import GameAPI

from serpent.frame_grabber import FrameGrabber

import serpent.cv

import skimage.util
import skimage.measure

import numpy as np

import time

########################
import serpent.ocr
import skimage.transform
from scipy.misc import imread
from scipy.linalg import norm
from scipy import sum, average
#########################

##########################################################
from serpent.game_api import GameAPI

from serpent.sprite import Sprite

from serpent.input_controller import KeyboardKey

import serpent.cv
import serpent.ocr

import numpy as np

import skimage.util
import skimage.transform

import pytesseract
from PIL import Image

import time
import uuid

###########
from scipy.misc import imread
from scipy.linalg import norm
from scipy import sum, average
##########
##########################################################
class trexAPI(GameAPI):

    def __init__(self, game=None):
        super().__init__(game=game)

    def my_api_function(self):
        pass

    def _capture_game_over_image(self):
        game_frame = FrameGrabber.get_frames([0]).frames[0]

        return serpent.cv.extract_region_from_image(
            game_frame.frame,
            self.game.screen_regions["GAME_OVER"]
        )


    def is_alive(self, game_frame, sprite_identifier):
        crashed_image = serpent.cv.extract_region_from_image(game_frame.frame, self.game.screen_regions["GAME_OVER"])
        crashed_sprite = Sprite("123", image_data=crashed_image[..., np.newaxis])

        sprite_name = sprite_identifier.identify(crashed_sprite, mode="CONSTELLATION_OF_PIXELS")

        
        #return False if sprite_name == "SPRITE_CRASHED" else True
        # if sprite_name == "SPRITE_TREX_GAME_OVER": #tjb
        #     print(f"## OVER") #tjb
        # else: #tjb
        #     print(f"## NOT OVER NOT OVER") #tjb
        
        # time.sleep(1) #tjb
        return False if sprite_name == "SPRITE_TREX_GAME_OVER" else True

    # class MyAPINamespace:

    #     @classmethod
    #     def my_namespaced_api_function(cls):
    #         api = trexAPI.instance
            
