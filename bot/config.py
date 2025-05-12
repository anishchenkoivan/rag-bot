from bot.searcher.frida_searcher import FridaSearcher
from bot.segmenter.nltk_segmenter import NltkSegmenter

token = ''
api_key = ""

default_segmenter = NltkSegmenter
default_searcher = FridaSearcher

class ImageRecognitionConfig:
    UseGPT = False
