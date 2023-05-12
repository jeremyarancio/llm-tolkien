import os
import logging
from typing import Mapping, Any

from torch import cuda
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling

import config
from training_utils import prepare_model, print_trainable_parameters


HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class LLMTolkien():

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.device = 'cuda' if cuda.is_available() else 'cpu'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer = tokenizer

    def train(
            self, 
            model_name: str,
            hf_repo: str, 
            lora_config: Mapping[str, Any],
            trainer_config: Mapping[str, Any],
            mlm: bool,
        ) -> None:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True)
        model = prepare_model(model)
        model = get_peft_model(model, LoraConfig(**lora_config))
        LOGGER.info(f"Model trainable parameters:\n {print_trainable_parameters(model)}")
        train_dataset = load_dataset(hf_repo, split="train")
        LOGGER.info(f"Train dataset downloaded:\n {train_dataset}")
        trainer = Trainer(
            model=model,
            args=TrainingArguments(**trainer_config),
            data_collator=DataCollatorForLanguageModeling(self.tokenizer, mlm=mlm),
        )
        model.config.use_cache = False  # silence warnings
        trainer.train()
        model.config.use_cache = True
        model.push_to_hub(hf_repo, token=HUGGINGFACE_TOKEN, private=True)

    def evaluate():
        pass

    def generate():
        pass


if __name__ == "__main__":

    lora_config = {
        "r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout, 
        'bias': config.lora_bias,
        "task_type": config.lora_task_type,
    }

    trainer_config = {
        "per_device_train_batch_size": config.per_device_train_batch_size, 
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "warmup_steps": config.warmup_steps, 
        "max_steps": config.max_steps, 
        "learning_rate": config.learning_rate, 
        "fp16": config.fp16,
        "logging_steps": config.logging_steps, 
        "output_dir": config.output_dir
    }

    model = LLMTolkien(config.model_name)
    model.train(
        
    )