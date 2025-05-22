from nltk.tokenize import sent_tokenize

from .segmenter import Segmenter


class NltkSegmenter(Segmenter):
    def __init__(self, data: str):
        super().__init__(data)

    def split(self) -> list[str]:
        return sent_tokenize(self.data)
