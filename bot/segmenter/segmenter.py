class Segmenter:
    def __init__(self, data: str):
        self.data = data

    def split(self) -> list[str]:
        raise NotImplementedError()
