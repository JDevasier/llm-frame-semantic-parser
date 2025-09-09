from unsloth import FastLanguageModel
import pandas as pd
from datasets import Dataset
import os

max_seq_length = 5020
model, tokenizer = FastLanguageModel.from_pretrained(
    # model_name="llama-3.1-70b-fsp-ft",
    model_name="qwen-2.5-7b-fsp-ft-cand",
    # model_name="qwen-2.5-7b-fsp-ft",
    max_seq_length=max_seq_length,
    dtype=None,
    load_in_4bit=True
)

def formatting_prompts_func_test(examples):
    convos = examples["messages"]
    texts = [tokenizer.apply_chat_template(convo[:-1], tokenize = False, add_generation_prompt = True) for convo in convos]
    return { "text" : texts, }

test_prompts_df = pd.read_json('../fn1.7-test-lex-prompts.jsonl', lines=True)
test_prompts = Dataset.from_pandas(test_prompts_df)

test_dataset = test_prompts.map(formatting_prompts_func_test, batched = True,)

FastLanguageModel.for_inference(model) # Enable native 2x faster inference

predictions = []


# Inference
# Start from the last prediction if the file exists
for i, example in enumerate(test_dataset['messages'][len(predictions):]):
    inputs = tokenizer.apply_chat_template(
        example[:-1],
        tokenize = True,
        add_generation_prompt = True, # Must add for generation
        return_tensors = "pt",
    ).to("cuda")
    outputs = model.generate(input_ids = inputs, use_cache = True, stop_strings=["}\n```\n"], eos_token_id = tokenizer.eos_token_id, tokenizer=tokenizer)
    # Decode the generated text, add to predictions, ignore prompt in output
    output_text = tokenizer.decode(outputs[0]).split('<|im_start|>assistant')[-1]
    predictions.append(output_text)

    # Write output to file
    if i % 10 == 0:
        pd.DataFrame(predictions, columns=["predictions"]).to_json("fn1.7-test-lex-predictions-temp.jsonl", lines=True, orient="records")


# Save the predictions
test_prompts_df["predictions"] = predictions
test_prompts_df.to_json("fn1.7-test-lex-predictions.jsonl", lines=True, orient="records")
