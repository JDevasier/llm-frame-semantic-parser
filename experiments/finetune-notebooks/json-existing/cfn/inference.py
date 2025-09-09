from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from trl import SFTTrainer, apply_chat_template
from peft import LoraConfig
from transformers import TrainingArguments
# from unsloth.chat_templates import get_chat_template
# from unsloth.chat_templates import train_on_responses_only
from datasets import Dataset
import pandas as pd
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description='Inference.')
parser.add_argument('--model', type=str, default='microsoft/phi-4', help='Model name or path')

args = parser.parse_args()

def count_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


tokenizer = AutoTokenizer.from_pretrained(args.model, device_map="auto", torch_dtype="auto")
model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype="auto")


cfn_train_prompts_df = pd.read_json('cfn-train-prompts.jsonl', lines=True)
cfn_test_prompts_df = pd.read_json('cfn-test-prompts.jsonl', lines=True)
en_train_prompts_df = pd.read_json('../fn1.7-train-prompts.jsonl', lines=True)

# train_prompt_compl_dataset = cfn_train_prompts_df.messages.apply(lambda x: {'prompt': x[:-1], 'completion': [x[-1]]})
# train_dataset = Dataset.from_pandas(pd.DataFrame(train_prompt_compl_dataset.values.flatten().tolist()))
# train_dataset = train_dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})

# en_train_prompt_compl_dataset = en_train_prompts_df.messages.apply(lambda x: {'prompt': x[:-1], 'completion': [x[-1]]})
# en_train_dataset = Dataset.from_pandas(pd.DataFrame(en_train_prompt_compl_dataset.values.flatten().tolist()))
# en_train_dataset = en_train_dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})

test_prompt_compl_dataset = cfn_test_prompts_df.messages.apply(lambda x: {'prompt': x[:-1], 'completion': [x[-1]]})
test_dataset = Dataset.from_pandas(pd.DataFrame(test_prompt_compl_dataset.values.flatten().tolist()))
test_dataset = test_dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer, })
test_dataset = test_dataset.remove_columns("completion")

peft_config = LoraConfig(
    r=16, lora_alpha=16, lora_dropout=0, bias="none", task_type="CAUSAL_LM", target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",]
)


print(f"Number of trainable parameters: {count_trainable_parameters(model)}")

# trainer.train()

# trainer.save_model("cfn-phi4-chinese-only")
# trainer.save_model("cfn-phi4-eng-then-chinese")
# trainer.save_model("cfn-phi4-eng-and-chinese")

# Save the model
# model.save_pretrained("cfn-phi4-chinese-only")
# model.save_pretrained("cfn-phi4-eng-then-chinese")
# model.save_pretrained("cfn-phi4-eng-and-chinese")

# Predict on test set
model.eval()

predictions = []

for i in range(len(test_dataset)):
    print(f"Predicting {i} / {len(test_dataset)}")
    inputs = tokenizer(test_dataset[i]['prompt'], return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=512)
    predictions.append(tokenizer.decode(outputs[0]))

    pd.DataFrame(predictions).to_csv(f"{os.path.basename(args.model)}-preds.csv")