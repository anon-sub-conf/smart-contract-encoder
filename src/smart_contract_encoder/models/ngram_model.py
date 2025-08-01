import pickle
from sentence_transformers import util
from sklearn.feature_extraction.text import CountVectorizer

class CardData:
    def set_evaluation_metrics(self, a, b, c, d):
        pass

class NgramEncoder:

    def __init__(self, load: bool = True):
        self.similarity_fn_name = 'cosine'
        self.model_card_data = CardData()
        self.trained = False
        if load:
            self._model = CountVectorizer(ngram_range=(5,5), dtype=float)
            # self.model = pickle.load(open("ngram_model.pkl", "rb"))
        else:
            self._model = CountVectorizer(ngram_range=(5,5), dtype=float)

    def encode(self, dataset, **args):
        if self.trained:
            return self._model.transform(dataset).toarray()
        else:
            self.trained = True
            self._model = self._model.fit(dataset)
            return self._model.transform(dataset).toarray()

    def encode_query(self, dataset, **args):
        if self.trained:
            return self._model.transform(dataset).toarray()
        else:
            self.trained = True
            self._model = self._model.fit(dataset)
            return self._model.transform(dataset).toarray()

    def encode_document(self, dataset, **args):
        if self.trained:
            return self._model.transform(dataset).toarray()
        else:
            self.trained = True
            self._model = self._model.fit(dataset)
            return self._model.transform(dataset).toarray()

    def train(self, dataset):
        self._model = self._model.fit(dataset)
        pickle.dump(self._model, open("ngram_model.pkl", "wb")) #TODO

    @property
    def model(self):
        return self

    @staticmethod
    def similarity(emb1, emb2):
        return util.cos_sim(emb1, emb2)
