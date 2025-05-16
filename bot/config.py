import os

from searcher.frida_searcher import FridaSearcher
from segmenter.nltk_segmenter import NltkSegmenter


telegram_token = os.getenv("TELEGRAM_TOKEN")
api_key = os.getenv("API_KEY")

default_segmenter = NltkSegmenter
default_searcher = FridaSearcher

class ImageRecognitionConfig:
    UseGPT = False

class SearcherConfig:
    UseGPT = True
