from config import default_segmenter, default_searcher


class Session:
    def __init__(self, data: str):
        self.segmenter = default_segmenter(data)
        self.searcher = default_searcher(self.segmenter.split())
        self.questions = []

    def add_questions(self, questions: list[str]):
        self.questions.extend(questions)

    def search(self):
        answers = self.searcher.retrieve_answers(self.questions)
        self.questions = []
        return answers
