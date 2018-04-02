from serpent.game import Game

from .api.api import trexAPI

from serpent.utilities import Singleton




class SerpenttrexGame(Game, metaclass=Singleton):

    def __init__(self, **kwargs):
        kwargs["platform"] = "executable"

        kwargs["window_name"] = "T-Rex Rush"

        
        
        kwargs["executable_path"] = "python ./main.py"
        
        

        super().__init__(**kwargs)

        self.api_class = trexAPI
        self.api_instance = None

    @property
    def screen_regions(self):
        regions = {
            "GAME_OVER": (80, 285, 108, 317),
            "SCORE_AREA": (16, 534, 29, 590),
            "HIGH_SCORE_AREA": (16, 468, 29, 524)
        }

        return regions

    @property
    def ocr_presets(self):
        presets = {
            "SAMPLE_PRESET": {
                "extract": {
                    "gradient_size": 1,
                    "closing_size": 1
                },
                "perform": {
                    "scale": 10,
                    "order": 1,
                    "horizontal_closing": 1,
                    "vertical_closing": 1
                }
            }
        }

        return presets
