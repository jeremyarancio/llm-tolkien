import os
import json
import logging
from pathlib import Path
from typing import Generator, Callable, Mapping

from datasets import Dataset
from transformers import AutoTokenizer

import config


HUGGINGFACE_TOKEN = os.getenv('HUGGINGFACE_TOKEN')
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_datasets(dataset_path: Path, min_length: int, context_length: int, 
                    test_size: float, shuffle: bool, num_grouped_pages: int) -> None:
    """Prepare dataset for training.
    Push the dataset to the hub.
    """
    tokenizer =  AutoTokenizer.from_pretrained(config.model_name)
    LOGGER.info(f'Start preparing dataset from {dataset_path}')
    texts = import_extracted_text(dataset_path=dataset_path, min_length=min_length, num_grouped_pages=num_grouped_pages)
    dataset = Dataset.from_dict({'text': list(texts)})
    dataset_dict = dataset.train_test_split(test_size=test_size, shuffle=shuffle)
    LOGGER.info(f'The dataset is composed  of {dataset_dict.num_rows} pages.')
    tokenized_dataset = dataset_dict.map(tokenize, batched=True, fn_kwargs={'tokenizer': tokenizer, 'context_length': context_length},
                                         remove_columns=dataset_dict["train"].column_names)
    LOGGER.info(f'The tokenized dataset is composed of {tokenized_dataset.num_rows} elements, each one composed of {context_length} tokens.')
    tokenized_dataset.push_to_hub("llm-tolkien", token=HUGGINGFACE_TOKEN, private=True)


def import_extracted_text(dataset_path: Path, min_length: int, num_grouped_pages: int) -> Generator[str, None, None]:
    """Prepare dataset for training from the jsonl file.

    Args:
        dataset_path (Path): Extracted text from the book
        min_length (int): Filter pages without text
        num_grouped_pages (int): Number of pages to group together. Because we drop the last tokens that are shorter 
        than max_length, we need to group pages together to have a bigger batch.

    Yields:
        Generator[str, None, None]: text of the pages
    """
    with open(dataset_path, 'r') as f:
        grouped_text = ""
        for num_page, line in enumerate(f, start=1):
            elt = json.loads(line)
            text = list(elt.values())[0]
            if len(text) > min_length:
                text = process_text(text)
                grouped_text += text
                if num_page % num_grouped_pages == 0:
                    yield grouped_text
                    grouped_text = ""


def process_text(text: str) -> str:
    text = text.replace('\n', ' ')
    return text


def tokenize(element: Mapping, tokenizer: Callable, context_length: int) -> str:
    inputs = tokenizer(element['text'], truncation=True, return_overflowing_tokens=True, 
                       return_length=True, max_length=context_length)
    inputs_batch = []
    for length, input_ids in zip(inputs['length'], inputs['input_ids']):
        if length == context_length: # We drop the last input_ids that are shorter than max_length
            inputs_batch.append(input_ids)
    return {"input_ids": inputs_batch}


if __name__ == '__main__':

    create_datasets(
        dataset_path=config.extraction_path, 
        min_length=config.min_length,
        context_length=config.context_length,
        test_size=config.test_size,
        shuffle=config.shuffle,
        num_grouped_pages=config.num_grouped_pages
    )