from sentence_transformers import SentenceTransformer
import torch

import config
from service.api_gateway import gemini_api_call
from service.data_extractor import format_questions, format_answers
from .searcher import Searcher


class FridaSearcher(Searcher):
    def __init__(self, segments: list[str]):
        super().__init__(segments)

    def retrieve_answers(self, questions, limit=2) -> list[str]:
        print("retrieving data")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(device)
        print("Segments: ", self.segments)
        print("Questions: ", questions)

        if len(self.segments) == 1:
            return [self.segments[0]] # Otherwise funny things happen in torch

        model = SentenceTransformer("ai-forever/FRIDA", device=device)
        search_docs = [f"search_document: {seg}" for seg in self.segments]
        search_query = [f"search_query: {q}" for q in questions]
        all_inputs = search_docs + search_query

        embeddings = model.encode(all_inputs, convert_to_tensor=True)
        doc_embeddings = embeddings[:len(self.segments)]
        query_embeddings = embeddings[len(self.segments):]

        answers = []
        for query_embedding in query_embeddings:
            sim_scores = (query_embedding @ doc_embeddings.T).squeeze(0)
            _, topk_indices = torch.topk(sim_scores, k=limit)
            top_segments = ";".join([self.segments[i] for i in topk_indices])
            answers.append(top_segments)

        if config.SearcherConfig.UseGPT:
            answers = self._format_answers(answers, questions)

        return answers

    def _format_answers(self, answers, questions):
        prompt = '''You are given a list of questions and then a list of answer to each question. Rephrase the answer so that it answers the questions properly and stylistically. Do not hallucinate or make up any information. Return answer as a string with no other symbols. separated by newline'''
        prompt += format_questions(questions)
        prompt += '\n'
        prompt += format_answers(answers)

        formatted_answers = gemini_api_call([prompt])
        return list(filter(lambda x: len(x) > 0, formatted_answers[0].split('\n')))
