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
        # load saved embeddings, due to python/library version incosistencies with SmartEmbed
        self.embeddings = pd.read_csv("./data/SmartEmbed_embeddings.csv")
        self.dataset = None

    def encode(self, dataset, **args):
        return self.encode_query(dataset)

    def encode_query(self, dataset, **args):
        res = []
        for q in dataset:
            index  = self.dataset.loc[self.dataset['func_code'] == q].index[0]
            res.append(self.embeddings.iloc[index].values[0][0])
        return np.array(res)

    def encode_document(self, dataset, **args):
        return self.encode_query(dataset, **args)

    @property
    def model(self):
        return self

    @staticmethod
    def similarity(emb1, emb2):
        return util.cos_sim(emb1, emb2)
