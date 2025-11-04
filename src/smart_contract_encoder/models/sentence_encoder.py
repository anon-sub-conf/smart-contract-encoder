from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.evaluation import TranslationEvaluator
from sentence_transformers.training_args import BatchSamplers
from smart_contract_encoder.models.base_model import BaseModel
from smart_contract_encoder.utils import *
from pathlib import Path

import pandas as pd
from sentence_transformers import util

class CardData:
    def set_evaluation_metrics(self, a, b, c, d):
        pass

def args_to_name(input_level: str = None, input_type: str = None,
                batch_size: int = 32,  save_steps: int = 3000, eval_steps: int = 50,
                logging_steps: int = 50):
    if input_level not in INPUT_LEVEL:
        raise Exception(f'Input must be one of {", ".join(INPUT_LEVEL)}')
    if input_type not in INPUT_TYPE:
        raise Exception(f'Input must be one of {", ".join(INPUT_TYPE)}')
    return f'{input_level}_{input_type}_{batch_size}_{save_steps}_{eval_steps}_{logging_steps}'

def args_to_path(input_level: str = None, input_type: str = None,
                batch_size: int = 32,  save_steps: int = 3000, eval_steps: int = 50,
                logging_steps: int = 50):
    subdir = args_to_name(input_level, input_type, batch_size, save_steps, eval_steps,
                          logging_steps)
    path = Path.joinpath(MODEL_DIR, subdir)
    return path

class SentenceEncoder(BaseModel):

    def __init__(self, load: bool = False, model_to_load: str = None, input_level: str = None, input_type: str = None,
                batch_size: int = 32,  save_steps: int = 3000, eval_steps: int = 50,
                logging_steps: int = 50):
        self.input_level = input_level
        self.input_type = input_type
        self.batch_size = batch_size
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.logging_steps = logging_steps
        self.similarity_fn_name = 'cosine'
        self.model_card_data = CardData()
        if load:
            if input_level != None:
                name = args_to_name(input_level, input_type, batch_size, save_steps, eval_steps,
                                    logging_steps)
                self.save_path = f'{model_to_load}_{name}'
                print(self.save_path)
        else:
            if input_level != None:
                self.save_path = args_to_path(input_level, input_type, batch_size, save_steps, eval_steps,
                                    logging_steps)
                print(self.save_path)
        if load:
            self._model = SentenceTransformer(model_to_load, device="cuda")
        else:
            self._model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device="cuda")

    def encode(self, dataset: pd.DataFrame):
        return self._model.encode(dataset)

    def encode_query(self, dataset, **args):
        print(dataset)
        return self.encode(dataset)

    def encode_document(self, dataset, **args):
        return self.encode(dataset)

    @property
    def model(self):
        return self

    @staticmethod
    def similarity(emb1, emb2):
        return util.cos_sim(emb1, emb2)

    def finetune_pairs(self, eval_dataset, train_dataset):
        loss = MultipleNegativesRankingLoss(self._model)
        args = SentenceTransformerTrainingArguments(
            output_dir=MODEL_DIR,
            num_train_epochs=1,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            warmup_ratio=0.1,
            fp16=True,  # Set to False if GPU can't handle FP16
            bf16=False,  # Set to True if GPU supports BF16
            batch_sampler=BatchSamplers.NO_DUPLICATES,
            eval_strategy="steps",
            save_only_model=1,
            eval_steps=self.eval_steps,
            save_strategy="steps",
            save_steps=self.save_steps,
            save_total_limit=2,
            logging_steps=self.logging_steps,
        )
        evaluator = TranslationEvaluator(
            source_sentences=eval_dataset["anchor"],
            target_sentences=eval_dataset["positive"],
        )
        trainer = SentenceTransformerTrainer(
            model=self._model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=loss,
            evaluator=evaluator,
        )
        trainer.train()
        self._model.save_pretrained(str(self.save_path))
        hist =  pd.DataFrame(trainer.state.log_history)
        hist.to_pickle(f"{str(self.save_path)}.pkl")

