import pandas as pd

train_data = pd.read_json('fn1.7-train-prompts.jsonl', lines=True)

train_data.sample(frac=0.01, random_state=0).to_json('fn1.7-train-prompts-01.jsonl', orient='records', lines=True)
train_data.sample(frac=0.05, random_state=0).to_json('fn1.7-train-prompts-05.jsonl', orient='records', lines=True)
train_data.sample(frac=0.1, random_state=0).to_json('fn1.7-train-prompts-10.jsonl', orient='records', lines=True)
train_data.sample(frac=0.25, random_state=0).to_json('fn1.7-train-prompts-25.jsonl', orient='records', lines=True)
train_data.sample(frac=0.5, random_state=0).to_json('fn1.7-train-prompts-50.jsonl', orient='records', lines=True)
train_data.sample(frac=0.75, random_state=0).to_json('fn1.7-train-prompts-75.jsonl', orient='records', lines=True)

