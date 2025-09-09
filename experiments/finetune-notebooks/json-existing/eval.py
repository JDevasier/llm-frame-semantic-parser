import pandas as pd
import re

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--input_file', type=str, required=True)
args = parser.parse_args()

def clean_output(output):
    cleaned_output = output
    # Extract XML code from the output
    json_code = re.search(r'```json(.*?)```', output, re.DOTALL)
    cleaned_output = ''
    if json_code:
        cleaned_output = json_code.group(1).strip()
    
    # Parse the JSON
    try:
        # cleaned_output = json.loads(cleaned_output)
        cleaned_output = eval(cleaned_output)
    except:
        return {}
    
    return cleaned_output

predictions = pd.read_json(args.input_file, lines=True)

predictions['output'] = predictions.apply(lambda x: clean_output(x['messages'][-1]['content']), axis=1)
predictions['prediction'] = predictions.apply(lambda x: clean_output(x['predictions']), axis=1)

tp = 0
near_miss = 0
fp = 0
fn = 0
tn = 0 # not really used because tags only have positive values, technically all other FEs are TN, but we don't really care about that (maybe we should?)

for label, pred, fes in predictions[['output', 'prediction', 'output']].values:

    # Get each predicted FE span from the prediction
    pred_tags = pred
    real_tags = fes

    # Check each predicted FE span
    for tag, content in pred_tags.items():
        if tag.capitalize() in real_tags:
            if content == real_tags[tag.capitalize()]:
                tp += 1
            else:
                # print(f'Near miss: {content} vs {real_tags[tag.capitalize()]}')
                fp += 1
                near_miss += 1
        else:
            fp += 1
    
    # Check each real FE span
    for tag, content in real_tags.items():
        if tag not in pred_tags:
            fn += 1
        elif content != pred_tags[tag]:
            fn += 1

# print(f'Perfect: {perfect} out of {len(predictions)}')
print(f'TP: {tp}')
print(f'Missed: {near_miss}')
print(f'FP: {fp}')
print(f'FN: {fn}')

print(f'Precision: {tp / (tp + fp)}')
print(f'Precision (ignoring near misses): {tp / (tp + fp - near_miss)}')
print(f'Recall: {tp / (tp + fn)}')
print(f'F1: {2 * tp / (2 * tp + fp + fn)}')

print(f'Accuracy: {tp / (tp + fn + fp)}')

