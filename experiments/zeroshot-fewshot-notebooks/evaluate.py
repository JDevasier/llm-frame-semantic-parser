import pandas as pd
import json
import re
import json_repair

def get_xml_tags(text: str) -> dict:
    # example: 'Your gift to Goodwill is important because <Cognizer>people with physical and Mental disabilities</Cognizer> sometimes need an extra hand to <Entity>know</Entity> the pride that comes with work .'
    # output: {'Cognizer': 'people with physical and Mental disabilities', 'Entity': 'know'}
    tags = {}

    # get all text between tags
    matches = re.findall(r'<(.*?)>(.*?)</(.*?)>', text)
    for match in matches:
        tag = match[0]
        content = match[1]
        tags[tag] = content

    return tags


# Extract FEs from the output
def get_markdown_tags(markdown: str) -> dict:
    # print(markdown)

    if '```markdown' in markdown:
        markdown = markdown.split('```markdown')[-1].split('```')[0].strip()
    elif '```' in markdown:
        markdown = markdown.split('```')[-2].strip()
        # print(markdown)

    clean_output = {}
    
    for line in markdown.split("\n"):
        clean_line = line.strip()
        if clean_line.startswith('-'):
            line_parts = clean_line.split(':')
            if len(line_parts) < 2:
                continue
            fe_name = line_parts[0].strip().replace('-', '').strip()
            fe_value = line_parts[1].strip()
            clean_output[fe_name] = fe_value
            
    return clean_output


def extract_json_from_output(output):
    try:
        json_part = "{}"
        if '```json' in output:
            json_part = output.split('```json')[-1].split('```')[0].strip()
        else:
            # Try regex to find JSON-like structure
            # match = re.search(r'\{.*?\}', output, re.DOTALL)
            # if match:
            #     json_part = match.group(0)
            json_part = output.split('{', 1)[-1].rsplit('}', 1)[0].strip()
        
        repaired_json = json_repair.loads(json_part)

        if isinstance(repaired_json, list):
            return repaired_json[0] if isinstance(repaired_json[0], dict) else {}
        return repaired_json if isinstance(repaired_json, dict) else {}
    except:
        return {}

import argparse
parser = argparse.ArgumentParser(description='Evaluate predictions against frame elements')
parser.add_argument('file_name', type=str, help='Path to the predictions file')
args = parser.parse_args()

file_name = args.file_name

if 'json-exist' in file_name:
    prompts = pd.read_csv('json-existing/fn1.7-test-prompts.csv')
elif 'json-complete' in file_name:
    prompts = pd.read_csv('json-complete/fn1.7-test-prompts.csv')
elif 'xml' in file_name:
    prompts = pd.read_csv('xml-tag/fn1.7-test-prompts.csv')
elif 'markdown' in file_name:
    prompts = pd.read_csv('markdown/fn1.7-test-prompts.csv')
else:
    raise ValueError("Unknown file type for prompts")


with open(file_name) as f:
    predictions = f.readlines()

predictions = [json.loads(p.strip()) for p in predictions]

prompts['prediction'] = predictions
predictions = prompts[['output', 'prediction', 'frame_elements']].copy()


if 'xmltag' in file_name:
    predictions['prediction'] = predictions.prediction.apply(lambda x: get_xml_tags(x['output']))
elif '-json' in file_name:
    predictions['prediction'] = predictions.prediction.apply(lambda x: extract_json_from_output(x['output']))
elif 'markdown' in file_name:
    predictions['prediction'] = predictions.prediction.apply(lambda x: get_markdown_tags(x['output']))
else:
    raise ValueError("Unknown file type for predictions")


tp = 0
near_miss = 0
fp = 0
fn = 0
tn = 0 # not really used because tags only have positive values, technically all other FEs are TN, but we don't really care about that (maybe we should?)

for label, pred, fes in predictions[['output', 'prediction', 'frame_elements']].values:
    # if len(pred) == 0:
    #     continue

    # Get each predicted FE span from the prediction
    pred_tags = pred
    # print(pred_tags)
    real_tags = eval(fes)

    # print(pred_tags)

    # Check each predicted FE span
    for tag, content in pred_tags.items():
        tag = tag.strip()

        if tag.strip().capitalize() in real_tags:
            if content == real_tags[tag.capitalize()]:
                # print(f'Perfect match: `{real_tags[tag.capitalize()]}` - `{content}`')
                tp += 1
            else:
                # print(f'Near miss: `{real_tags[tag.capitalize()]}` - `{content}`')
                fp += 1
                near_miss += 1
        else:
            if content != '':
                fp += 1
    
    # Check each real FE span
    for tag, content in real_tags.items():
        if content == '':
            continue

        if tag not in pred_tags:
            fn += 1
            # print(f'FN: `{tag}` - `{content}`')

        elif content != pred_tags[tag]:
            fn += 1
            # print(f'FN: `{tag}` - `{content}` (predicted: `{pred_tags[tag]}`)')

# print(f'Perfect: {perfect} out of {len(predictions)}')
print(f'TP: {tp}')
# print(f'Missed: {near_miss}')
print(f'FP: {fp}')
print(f'FN: {fn}')

print(f'Precision: {tp / (tp + fp):0.3f}')
# print(f'Precision (ignoring near misses): {tp / (tp + fp - near_miss):0.3f}')
print(f'Recall: {tp / (tp + fn):0.3f}')
print(f'F1: {2 * tp / (2 * tp + fp + fn):0.3f}')

print(f'Accuracy: {tp / (tp + fn + fp):0.3f}')

