from typing import List
import spacy
from joblib import load


class QulogSVC:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_trf')
        self.model = self.load('./model')


    def load(self, model_file_path):
        return load(model_file_path)


    def _get_trf_embedding(self, doc):
        return doc._.trf_data.tensors[-1].ravel()


    def _get_trf_embeddings(self, trf_docs):
        embeddings = []
        for doc in trf_docs:
            embeddings.append(self._get_trf_embedding)
        return embeddings


    def predict(self, log_line: str):
        doc = self.nlp(log_line)
        embedding = self._get_trf_embedding(doc)
        return self.model.predict(embedding)


    def predict_batch(self, log_lines: List[str]):
        docs = list(self.nlp.pipe(self.log_messages))

