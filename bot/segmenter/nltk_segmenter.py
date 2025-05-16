from nltk.tokenize import sent_tokenize
import nltk

from .segmenter import Segmenter


class NltkSegmenter(Segmenter):
    def __init__(self, data: str):
        nltk.download('punkt_tab')
        super().__init__(data)

    def split(self) -> list[str]:
        return sent_tokenize(self.data)
