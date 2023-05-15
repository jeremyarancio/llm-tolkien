import os
from pathlib import Path

REPO_DIR = Path(os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))

# Text extraction
url = 'https://gosafir.com/mag/wp-content/uploads/2019/12/Tolkien-J.-The-lord-of-the-rings-HarperCollins-ebooks-2010.pdf'
header_height = 60  # Main text distance from the top of the page: to remove header
footer_height = 540 # Remove footer
start_page = 45
end_page = 1055
extraction_path = REPO_DIR / "llm/data/extracted_text.jsonl"

# Text processing
min_length = 100
num_grouped_pages = 10

# HF repo
hf_repo = "JeremyArancio/llm-tolkien"

# Dataset
context_length = 128
batch_size = 1000
test_size = 0.1
shuffle = True

# Training
model_name = 'bigscience/bloom-3b'
lora_r = 16 # attention heads
lora_alpha = 32 # alpha scaling
lora_dropout = 0.05
lora_bias = "none"
lora_task_type = "CAUSAL_LM" # set this for CLM or Seq2Seq

## Trainer config
per_device_train_batch_size = 4 
gradient_accumulation_steps = 4 # Equivalent of doing a batch of 16
warmup_steps = 100 
num_train_epochs=3
weight_decay=0.1
learning_rate = 2e-4 
fp16 = True
logging_steps = 1
overwrite_output_dir = True
output_dir = REPO_DIR / 'llm/training_outputs'
evaluation_strategy = "epoch"
save_strategy = "epoch"
push_to_hub = True

## Data collator
mlm =False

## Generate
max_new_tokens = 50
temperature = 0.5
do_sample = False
