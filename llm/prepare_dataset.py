import logging
from pathlib import Path
import json
from typing import Generator, Callable, Mapping

from transformers import AutoTokenizer
from datasets import DatasetDict, Dataset

import config


LOGGER = logging.getLogger(__name__)


def create_datasets(dataset_path: Path, min_length: int, batch_size: int, 
                   context_length: int, test_size: float, shuffle: bool) -> None:
    """Prepare dataset for training.
    """
    tokenizer =  AutoTokenizer.from_pretrained(config.model_name)
    LOGGER.info(f'Start preparing dataset from {dataset_path}')
    texts = import_extracted_text(dataset_path=dataset_path, min_length=min_length)
    LOGGER.info(f'Finished preparing dataset from {dataset_path}')
    dataset = Dataset.from_dict({'text': list(texts)})
    dataset_dict = dataset.train_test_split(test_size=test_size, shuffle=shuffle)
    tokenized_dataset = dataset_dict.map(tokenize, batched=True, batch_size=batch_size, 
                                         fn_kwargs={'tokenizer': tokenizer, 'context_length': context_length})
    pass


def import_extracted_text(dataset_path: Path, min_length: int) -> Generator[str, None, None]:
    """Prepare dataset for training from the jsonl file.
    """
    with open(dataset_path, 'r') as f:
        for line in f:
            elt = json.loads(line)
            text = list(elt.values())[0]
            if len(text) > min_length:
                text = process_text(text)
                yield text


def process_text(text: str) -> str:
    text = text.replace('\n', ' ')
    return text


def tokenize(element: Mapping, tokenizer: Callable, context_length: int) -> str:
    inputs = tokenizer(element['text'], truncation=True, return_overflowing_tokens=True, 
                               return_length=True, max_length=context_length, return_tensors='pt')
    inputs_batch = []
    for length, input_ids in zip(inputs['length'], inputs['input_ids']):
        if length == context_length: # We drop the last input_ids that are shorter than max_length
            inputs_batch.append(input_ids)
    return {'input_ids': inputs_batch}


if __name__ == '__main__':

    create_datasets(
        dataset_path=config.extraction_path, 
        min_length=config.min_length,
        batch_size=config.batch_size,
        context_length=config.context_length,
        test_size=config.test_size,
        shuffle=config.shuffle
    )