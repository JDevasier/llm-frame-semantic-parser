import xml.etree.ElementTree as ET
import pandas as pd
import re
import os


# Function to clean up encoded HTML entities like &lt; &gt;
def clean_html(text):
    replacements = {
        '&lt;': '<',
        '&gt;': '>',
        '&amp;': '&',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def extract_examples(text):
    # Examples are enclosed in <ex> tags </ex>
    # Also, remove the examples from the original text, return both
    examples = re.findall(r'<ex>(.*?)</ex>', text, re.DOTALL)
    text = re.sub(r'<ex>.*?</ex>', '', text, flags=re.DOTALL)
    return text, examples

def remove_tags(text):
    # Remove all tags from the text
    return re.sub(r'<.*?>', '', text)

def remove_defroot(text):
    # Remove def-root open and close tags
    return re.sub(r'<def-root.*?>|</def-root>', '', text)

def remove_extags(text):
    # Remove all <ex> tags from the text
    return re.sub(r'<ex>.*?</ex>', '', text, flags=re.DOTALL)

def replace_t_tags(text, bold=False):
    # Remove all <t> tags from the text
    if bold:
        return re.sub(r'<(?:t|m)>(.*?)</(?:t|m)>', r'**\1**', text)
    return re.sub(r'<(?:t|m)>(.*?)</(?:t|m)>', r'\1', text)

def replace_tags(text):
    # Replace tags of the form: <fex name="Agent"> with <Agent> and replace closing tags from </fex> to </Agent>
    return re.sub(r'<(?:fex|m)\s+name="(.*?)">(.*?)</(?:fex|m)>', r'<\1>\2</\1>', text)

def replace_fen_tags(text):
    # Replace tags of form: <fen>FE</fen> with <FE>FE</FE>
    return re.sub(r'<fen>(.*?)</fen>', r'<\1>\1</\1>', text)

def clean_text(text, remove_tag=False):
    # Strip leading and trailing whitespaces
    # Replace multiple whitespaces with a single space
    clean = re.sub(r'\s+', ' ', remove_tags(text.strip()) if remove_tag else replace_tags(text.strip()))

    return remove_dni(clean)

def remove_dni(text):
    # Remove tags containing DNI
    return re.sub(r'<\w+>DNI<\/\w+>', '', text)
    
    
def replace_tag_abbrev(text, fe_abbr):
    # Replace FE abbreviations with their full names
    for abbr, name in fe_abbr.items():
        text = text.replace(f'<{abbr}>', f'<{name}>').replace(f'</{abbr}>', f'</{name}>')
    return text

def get_annotation_samples(data: pd.DataFrame, frame_name: str, num_samples: int = 5):
    frame_samples = data.query(f'frame == "{frame_name}"')
    frame_samples = frame_samples.drop_duplicates('text').sort_values(by='fe', key=lambda x: x.str.len(), ascending=False)
    
    return frame_samples.head(num_samples)

def add_tags(text, fes):
    sorted_fes = sorted(fes, key=lambda x: x[1])
    
    tagged_str = ''
    prev_end = 0
    for fe in sorted_fes:
        fe_name, start, end = fe
        tagged_str += text[prev_end:start] + f'<{fe_name}>' + text[start:end] + f'</{fe_name}>'
        prev_end = end
    tagged_str += text[prev_end:]
    
    return tagged_str
    

def get_definition(node):
    text, examples = extract_examples(node.text)
    
    text = text.split('\n')[0]

    return remove_defroot(clean_text(text, False)), [clean_text(ex) for ex in examples if ex.strip() != '']


# Function to extract frame information from the XML
def parse_frame(train_data, xml_file):
    # Parse the XML content
    root = ET.parse(xml_file).getroot()

    # Namespace handling
    ns = {'fn': 'http://framenet.icsi.berkeley.edu'}

    # Extract frame name and definition
    frame_name = root.attrib.get('name')
    frame_definition, examples = get_definition(root.find('fn:definition', ns))

    anno_examples = get_annotation_samples(train_data, frame_name, 5)

    tagged_annos = anno_examples.apply(lambda x: add_tags(x['text'], x['fe']), axis=1) if len(anno_examples) > 0 and anno_examples['fe'].iloc[0] else pd.Series()

    examples = examples + tagged_annos.tolist()

    # Extract frame elements
    frame_elements = []
    fe_abbr = {}
    for fe in root.findall('fn:FE', ns):
        fe_name = fe.attrib.get('name')
        fe_definition, fe_examples = get_definition(fe.find('fn:definition', ns))
        
        core_type = fe.attrib.get('coreType')
        
        # Extract frame element abbreviations
        abbr = fe.attrib.get('abbrev')
        if abbr:
            # if abbr in fe_abbr:
            #     assert fe_abbr[abbr] == fe_name, f"Abbreviation {abbr} already exists for {fe_abbr[abbr]} and {fe_name}"
            fe_abbr[fe.attrib.get('abbrev')] = fe_name
            
        frame_elements.append({
            'name': fe_name,
            'definition': remove_tags(fe_definition),
            'examples': [replace_tag_abbrev(replace_t_tags(x, True), fe_abbr) for x in fe_examples],
            'core_type': core_type
        })

    # Return extracted data
    return {
        'frame_name': frame_name.strip(),
        'frame_definition': remove_tags(frame_definition).strip(),
        'examples': examples,
        'frame_elements': frame_elements
    }

def convert_tags_to_json(example: str, all_fes: set) -> dict:
    # Find all tags and their content in the example
    tags = re.findall(r'<(.*?)>(.*?)</\1>', example)
    dict_tags = dict(tags)
    
    dict_tags.pop('t', None)
    
    return {fe: dict_tags[fe] for fe in dict_tags if fe in all_fes}

def fill_missing_fes(example: dict, all_fes: set) -> dict:
    filled_example = example.copy()
    for fe in all_fes:
        if fe not in filled_example:
            filled_example[fe] = ''
    
    return filled_example

import pandas as pd
# Load training annotations

train_data = pd.read_json('../../data/raw/os_train.jsonl', lines=True)

# Parse the frame
frame_dir = '../../data/raw/fndata-1.7/frame/'
frame_strings = []
for frame_file in os.listdir(frame_dir):
# for frame_file in ['Abusing.xml']:
    if not frame_file.endswith('.xml'):
        continue
    
    print(frame_file)
    
    frame_data = parse_frame(train_data, os.path.join(frame_dir, frame_file))

    desc_str = ""
    desc_str += f"Frame Name: {frame_data['frame_name'].strip()}\n"
    desc_str += f"Frame Definition: {frame_data['frame_definition'].strip()}\n"
    
    # desc_str += "Examples:\n"
    # for ex in frame_data['examples']:
    #     if ex.strip():
    #         ex_json = json.dumps(convert_tags_to_json(replace_t_tags(ex.strip()), set([fe['name'] for fe in frame_data['frame_elements']])))
    #         if ex_json and ex_json != '{}':
    #             desc_str += f"  - {remove_tags(ex.strip()).strip()} -> {ex_json}\n"

    desc_str += "\nFrame Elements:\n"
    for fe in frame_data['frame_elements']:
        desc_str += f"{fe['name'].strip()} ({fe['core_type'].strip()}): {fe['definition'].strip()}\n"
        # if fe['examples']:
        #     for ex in fe['examples']:
        #         ex_json = json.dumps(convert_tags_to_json(ex.strip(), set([fe['name'] for fe in frame_data['frame_elements']])))
        #         if ex_json and ex_json != '{}':
        #             desc_str += f"  - {remove_tags(ex.strip()).strip()} -> {ex_json}\n"
        # desc_str += "\n"

    frame_strings.append({'name': frame_data['frame_name'], 'description': desc_str})

df = pd.DataFrame(frame_strings)
df.to_csv('frame_descriptions_json.csv', index=False)

import pandas as pd

frame_descriptions = pd.read_csv('frame_descriptions_json.csv')

system_prompt = """### Task:
You are given a sentence and a frame with its associated frame elements and sometimes examples. Your task is to label the frame elements in the sentence using JSON. Keys should only be one of the defined frame elements. Do not make up your own frame elements, and do not remove or change the input in any way. Identify the frame elements based on the highlighted target word. 

### Notes:
- Return the tagged sentence in a ```json ``` code block.
- Texts must not overlap.
"""

user_prompt = """### Frame Information:
{frame_info}

### Input:
{input_sentence}
"""

assistant_prompt = """### Output: 
```json
{output_sentence}
```
"""



# Create dicts for each sample:
# - input_sentence w/ target span surrounded with ** for highlighting
# - frame_name
# - frame_elements (as text, not spans)

def get_json_output(text, frame_elements):
    sorted_fes = sorted(frame_elements.items(), key=lambda x: text.find(x[1]))
    
    sorted_fes = dict(sorted_fes)
    
    return sorted_fes

test_samples = []
frame_descriptions_dict = frame_descriptions.set_index('name').to_dict()['description']

for row in train_data.iterrows():
    # Index(['target', 'text', 'tokens', 'lu', 'frame', 'fe'], dtype='object')
    idx, data = row
    
    # Get input sentence
    input_sentence = data['text'][:data['target'][0]] + '**' + data['text'][data['target'][0]:data['target'][1]] + '**' + data['text'][data['target'][1]:]
    
    # Get frame name
    frame_name = data['frame']
    
    # Get frame elements
    frame_elements = {}
    for fe in data['fe']:
        frame_elements[fe[0]] = data['text'][fe[1]:fe[2]]
        
    # Get expected output
    expected_output = get_json_output(data['text'], frame_elements)

    sample = {
        'messages': [
            {
                'role': 'system',
                'content': system_prompt
            },
            {
                'role': 'user',
                'content': user_prompt.format(frame_info=frame_descriptions_dict[frame_name].strip(), input_sentence=input_sentence)
            },
            {
                'role': 'assistant',
                'content': assistant_prompt.format(output_sentence=expected_output)
            }
        ]
    }
    
    if len(frame_elements) > 0:
        test_samples.append(sample)


pd.DataFrame(test_samples).to_json('fn1.7-train-prompts.jsonl', orient='records', lines=True)

test_data = pd.read_json('../../data/raw/os_test.jsonl', lines=True)

# Create dicts for each sample:
# - input_sentence w/ target span surrounded with ** for highlighting
# - frame_name
# - frame_elements (as text, not spans)

def get_json_output(text, frame_elements):
    sorted_fes = sorted(frame_elements.items(), key=lambda x: text.find(x[1]))
    
    sorted_fes = dict(sorted_fes)
    
    return sorted_fes

test_samples = []
frame_descriptions_dict = frame_descriptions.set_index('name').to_dict()['description']

for row in test_data.iterrows():
    # Index(['target', 'text', 'tokens', 'lu', 'frame', 'fe'], dtype='object')
    idx, data = row
    
    # Get input sentence
    input_sentence = data['text'][:data['target'][0]] + '**' + data['text'][data['target'][0]:data['target'][1]] + '**' + data['text'][data['target'][1]:]
    
    # Get frame name
    frame_name = data['frame']
    
    # Get frame elements
    frame_elements = {}
    for fe in data['fe']:
        frame_elements[fe[0]] = data['text'][fe[1]:fe[2]]
        
    # Get expected output
    expected_output = get_json_output(data['text'], frame_elements)

    sample = {
        'messages': [
            {
                'role': 'system',
                'content': system_prompt
            },
            {
                'role': 'user',
                'content': user_prompt.format(frame_info=frame_descriptions_dict[frame_name].strip(), input_sentence=input_sentence)
            },
            {
                'role': 'assistant',
                'content': assistant_prompt.format(output_sentence=expected_output)
            }
        ]
    }
    
    if len(frame_elements) > 0:
        test_samples.append(sample)


pd.DataFrame(test_samples).to_json('fn1.7-test-prompts.jsonl', orient='records', lines=True)