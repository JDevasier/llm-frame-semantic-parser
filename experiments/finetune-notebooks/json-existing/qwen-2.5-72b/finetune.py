from unsloth import FastLanguageModel
import pandas as pd
from datasets import Dataset

from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported

from unsloth.chat_templates import get_chat_template

from unsloth.chat_templates import train_on_responses_only
import os

# work_dir = os.environ['WORK'] if 'WORK' in os.environ else os.getcwd()

train_prompts_df = pd.read_json('../fn1.7-train-prompts.jsonl', lines=True)
train_prompts = Dataset.from_pandas(train_prompts_df)

max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    # model_name="llama-3.1-70b-fsp-ft",
    model_name="unsloth/Qwen2.5-72B",
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 32, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)


tokenizer = get_chat_template(
    tokenizer,
    chat_template = "qwen-2.5",
)

def formatting_prompts_func(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }

def formatting_prompts_func_test(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo[:-1], tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }

dataset = train_prompts.map(formatting_prompts_func, batched = True,)

test_prompts_df = pd.read_json('../fn1.7-test-prompts.jsonl', lines=True)
test_prompts = Dataset.from_pandas(test_prompts_df)

test_dataset = test_prompts.map(formatting_prompts_func_test, batched = True,)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        # warmup_steps = 5,
        num_train_epochs = 1, # Set this for 1 full training run.
        # max_steps = 20,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "output-qwen-2.5-72b-fsp-ft-2",
        report_to = "none", # Use this for WandB etc
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|im_start|>user\n",
    response_part = "<|im_start|>assistant\n",
)

trainer_stats = trainer.train()

# Save the model
model.save_pretrained("qwen-2.5-72b-fsp-ft")
tokenizer.save_pretrained("qwen-2.5-72b-fsp-ft")

# FastLanguageModel.for_inference(model) # Enable native 2x faster inference

# predictions = []

# for i, example in enumerate(test_dataset['messages']):
#     inputs = tokenizer.apply_chat_template(
#         example[:-1],
#         tokenize = True,
#         add_generation_prompt = True, # Must add for generation
#         return_tensors = "pt",
#     ).to("cuda")
#     outputs = model.generate(input_ids = inputs, max_new_tokens = 256, use_cache = True)
#     # Decode the generated text, add to predictions, ignore prompt in output
#     predictions.append(tokenizer.decode(outputs[0]).split("<|start_header_id|>assistant<|end_header_id|>\n\n")[1])

# # Save the predictions
# pd.DataFrame(predictions).to_csv("predictions.csv", index = False, header = False)
