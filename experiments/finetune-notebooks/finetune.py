from datasets import Dataset, load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from peft import LoraConfig, get_peft_model

import argparse

parser = argparse.ArgumentParser(description="Fine-tune model")
parser.add_argument("--train_dataset", type=str, required=True)
parser.add_argument("--val_dataset", type=str)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--output_dir", type=str, default="finetuned_model")

args = parser.parse_args()

# Load the training and validation datasets
train_dataset = pd.read_json(args.train_dataset, lines=True) 

# Must use prompt-completion format
if 'prompt' in train_dataset.columns and 'completion' in train_dataset.columns:
    train_dataset = Dataset.from_pandas(train_dataset[['prompt', 'completion']])
elif 'messages' in train_dataset.columns:
    train_dataset = Dataset.from_pandas(train_dataset[['messages']])
else:
    raise ValueError("Training dataset must contain 'prompt' and 'completion' columns or 'messages' column")

if args.val_dataset:
    val_dataset = pd.read_json(args.val_dataset, lines=True)
    if 'prompt' in val_dataset.columns and 'completion' in val_dataset.columns:
        val_dataset = Dataset.from_pandas(val_dataset[['prompt', 'completion']])
    elif 'messages' in val_dataset.columns:
        val_dataset = Dataset.from_pandas(val_dataset[['messages']])
    else:
        raise ValueError("Validation dataset must contain 'prompt' and 'completion' columns or 'messages' column")
else:
    val_dataset = None

# Load the model
model = AutoModelForCausalLM.from_pretrained(args.model)
tokenizer = AutoTokenizer.from_pretrained(args.model, max_seq_length=4096)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

peft_config = LoraConfig(
    r = 16, # Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, 
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir=args.output_dir,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=500,
    # eval_strategy="steps",
    # eval_steps=500,
    save_total_limit=2,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_steps=10,
    fp16=True,
    # report_to="wandb",
    # run_name="finetune_run",
)

from transformers import DataCollatorForLanguageModeling

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    # data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors="pt"),
    # max_seq_length=4096
)

trainer.train()