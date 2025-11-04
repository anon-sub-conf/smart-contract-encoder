import pickle
from sentence_transformers import util
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np

class CardData:
    def set_evaluation_metrics(self, a, b, c, d):
        pass

class SmartEmbed:

    def __init__(self):
        self.similarity_fn_name = 'cosine'
        self.model_card_data = CardData()
        self.embeddings = pd.read_pickle("./data/SmartEmbed_embeddings.pkl")
        print(self.embeddings)
        self.dataset = None
        # self.trained = False
        # if load:
        #     self._model = CountVectorizer(ngram_range=(5,5), dtype=float)
        #     # self.model = pickle.load(open("ngram_model.pkl", "rb"))
        # else:
        #     self._model = CountVectorizer(ngram_range=(5,5), dtype=float)

    def encode(self, dataset, **args):
        # print(dataset)
        # print(self.dataset.)
        return self.encode_query(dataset)
        # pass
        # if self.trained:
        #     return self._model.transform(dataset).toarray()
        # else:
        #     self.trained = True
        #     self._model = self._model.fit(dataset)
        #     return self._model.transform(dataset).toarray()

    def encode_query(self, dataset, **args):
        res = []
        for q in dataset:
            index  = self.dataset.loc[self.dataset['func_code'] == q].index[0]
            # print(index)
            res.append(self.embeddings.iloc[index].values[0][0])
        # print(res)
        return np.array(res)
        # pass
        # if self.trained:
        #     return self._model.transform(dataset).toarray()
        # else:
        #     self.trained = True
        #     self._model = self._model.fit(dataset)
        #     return self._model.transform(dataset).toarray()

    def encode_document(self, dataset, **args):
        # print(dataset)
        return self.encode_query(dataset, **args)
        # pass
        # if self.trained:
        #     return self._model.transform(dataset).toarray()
        # else:
        #     self.trained = True
        #     self._model = self._model.fit(dataset)
        #     return self._model.transform(dataset).toarray()

    # def train(self, dataset):
    #     self._model = self._model.fit(dataset)
    #     pickle.dump(self._model, open("ngram_model.pkl", "wb")) #TODO

    @property
    def model(self):
        return self

    @staticmethod
    def similarity(emb1, emb2):
        return util.cos_sim(emb1, emb2)
