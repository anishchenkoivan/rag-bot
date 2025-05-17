from datetime import datetime

from sentence_transformers import SentenceTransformer
import torch
import gc

import config
from service.api_gateway import gemini_api_call
from service.data_extractor import format_questions, format_answers
from .searcher import Searcher


class FridaSearcher(Searcher):
    def __init__(self, segments: list[str]):
        super().__init__(segments)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer("ai-forever/FRIDA", device=device)

        search_docs = [f"search_document: {seg}" for seg in self.segments]
        self.doc_embeddings = self.model.encode(search_docs, convert_to_tensor=True)

    def retrieve_answers(self, questions, limit=2) -> list[str]:
        if len(self.segments) == 1:
            return [self.segments[0]] # Otherwise funny things happen in torch

        search_query = [f"search_query: {q}" for q in questions]
        query_embeddings = self.model.encode(search_query, convert_to_tensor=True)

        answers = []
        for query_embedding in query_embeddings:
            sim_scores = (query_embedding @ self.doc_embeddings.T).squeeze(0)
            _, topk_indices = torch.topk(sim_scores, k=limit)
            top_segments = ";".join([self.segments[i] for i in topk_indices])
            answers.append(top_segments)

        print(f"{datetime.now()} Response from Frida: {answers}")
        if config.SearcherConfig.UseGPT and answers:
            answers = self._format_answers(answers, questions)

        return answers

    def clear(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()

    def _format_answers(self, answers, questions):
        prompt = '''You are given a list of questions and then a list of answer to each question. Rephrase the answer so that it answers the questions properly and stylistically. Do not hallucinate or make up any information. Retain as much relevant information from original text as possible without breaking the context. Return answer as a string with no other symbols. Give an answer to each question. Separate answers to different questions with ||. If you only get one question, do not include || in the result'''
        prompt += format_questions(questions)
        prompt += '\n'
        prompt += format_answers(answers)

        formatted_answers = gemini_api_call([prompt])
        return list(filter(lambda x: x.strip(), formatted_answers[0].split('||')))
