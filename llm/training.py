import os
import logging
import argparse
from typing import Mapping, Any

from torch import cuda
from datasets import load_dataset
from peft import LoraConfig, PeftModel, PeftConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling

import config
from training_utils import prepare_model, print_trainable_parameters, compute_perplexity


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class LLMTolkien():

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.device = 'cuda' if cuda.is_available() else 'cpu'

    def train(
            self, 
            hf_repo: str, 
            lora_config: Mapping[str, Any],
            trainer_config: Mapping[str, Any],
            mlm: bool,
        ) -> None:
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto", load_in_8bit=True)
        model = prepare_model(model)
        model = get_peft_model(model, LoraConfig(**lora_config))
        LOGGER.info(f"Model trainable parameters:\n {print_trainable_parameters(model)}")
        dataset = load_dataset(hf_repo)
        LOGGER.info(f"Train dataset downloaded:\n {dataset['train']}")
        LOGGER.info(f"Number of tokens for the training: {dataset['train'].num_rows*len(dataset['train']['input_ids'][0])}")
        trainer = Trainer(
            model=model,
            train_dataset=dataset['train'],
            args=TrainingArguments(**trainer_config),
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=mlm)
        )
        model.config.use_cache = False  # silence warnings
        trainer.train()
        model.config.use_cache = True
        model.push_to_hub(repo_id=hf_repo)
        tokenizer.push_to_hub(repo_id=hf_repo)

    def evaluate():
        pass

    def generate(self, prompt: str, hf_repo: str, max_new_tokens: int, temperature: float, do_sample: bool) -> None:
        # Import the model
        config = PeftConfig.from_pretrained(hf_repo)
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, return_dict=True, load_in_8bit=True, device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
        # Load the Lora model
        self.model = PeftModel.from_pretrained(model, hf_repo)

        # Generate text
        inputs = tokenizer(prompt, return_tensors="pt")
        tokens = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )
        print(tokenizer.decode(tokens[0]))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train an LLM with Lora.")
    parser.add_argument("--model-name", type=str, default=config.model_name, help="The name of the model to train.")
    parser.add_argument("--hf-repo", type=str, default=config.hf_repo, help="The name of the HuggingFace repo to push the model to.")
    parser.add_argument("--mlm", type=bool, default=config.mlm, help="Whether to use MLM or not for the training.")
    parser.add_argument('--lora-r', type=float, default=config.lora_r, help="The Lora parameter r, the number of heads.")
    parser.add_argument('--lora-alpha', type=float, default=config.lora_alpha, help="Lora parameter.")
    parser.add_argument('--lora-dropout', type=float, default=config.lora_dropout, help="Lora dropout.")
    parser.add_argument('--lora-bias', type=str, default=config.lora_bias, help="Lora bias.")
    parser.add_argument('--lora-task-type', type=str, default=config.lora_task_type, help="Lora task type.")
    parser.add_argument('--per-device-train-batch-size', type=int, default=config.per_device_train_batch_size, help="The batch size per device for the training.")
    parser.add_argument('--gradient-accumulation-steps', type=int, default=config.gradient_accumulation_steps, help="The number of gradient accumulation steps.")
    parser.add_argument('--warmup-steps', type=int, default=config.warmup_steps, help="The number of warmup steps.")
    parser.add_argument('--weight-decay', type=float, default=config.weight_decay, help="The weight decay.")
    parser.add_argument('--num-train-epochs', type=float, default=config.num_train_epochs, help="The number of training epochs.")
    parser.add_argument('--learning-rate', type=float, default=config.learning_rate, help="The learning rate.")
    parser.add_argument('--fp16', type=bool, default=config.fp16, help="Whether to use fp16 or not.")
    parser.add_argument('--logging-steps', type=int, default=config.logging_steps, help="The number of logging steps.")
    parser.add_argument('--output-dir', type=str, default=config.hf_repo, help="The output directory.")
    parser.add_argument('--overwrite-output_dir', type=bool, default=config.overwrite_output_dir, help="Whether to overwrite the output directory.")
    parser.add_argument('--save-strategy', type=str, default=config.save_strategy, help="The saving strategy.")
    parser.add_argument('--evaluation-strategy', type=str, default=config.evaluation_strategy, help="The evaluation strategy.")
    parser.add_argument('--push-to-hub', type=bool, default=config.push_to_hub, help="Whether to push the model to the HuggingFace Hub.")
    args = parser.parse_args()


    lora_config = {
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout, 
        'bias': args.lora_bias,
        "task_type": args.lora_task_type,
    }

    trainer_config = {
        "per_device_train_batch_size": args.per_device_train_batch_size, 
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate, 
        "fp16": args.fp16,
        "logging_steps": args.logging_steps, 
        "output_dir": args.output_dir,
        "overwrite_output_dir": args.overwrite_output_dir,
        "evaluation_strategy": args.evaluation_strategy,
        "save_strategy": args.save_strategy,
        "push_to_hub": args.push_to_hub
    }

    model = LLMTolkien(args.model_name)
    model.train(
        hf_repo=args.hf_repo,
        lora_config=lora_config,
        trainer_config=trainer_config,
        mlm=args.mlm
    )
