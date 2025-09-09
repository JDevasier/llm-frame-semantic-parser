# Tested using Python 3.11 and requirements.txt provided
# NOTE: Not all packages in requirements.txt are used (most aren't) and I recommend starting with a 
# clean environment and only installing the necessary packages, i.e., 
# `pip install torch vllm transformers json_repair pandas spacy inflect nltk``
# Also consider `pip install flash-attn`` for much faster inference if your GPU supports it.

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer
from json_repair import repair_json

import pandas as pd
import json
import re

# NOTE: Our LLM is finetuned on the outputs only. While we haven't tested it, you may have different results if you modify the prompts.
SYSTEM_PROMPT = """### Task:
You are given a sentence and a frame with its associated frame elements and sometimes examples. Your task is to label the frame elements in the sentence using JSON. Keys should only be one of the defined frame elements. Do not make up your own frame elements, and do not remove or change the input in any way. Identify the frame elements based on the highlighted target word. 

### Notes:
- Return the tagged sentence in a ```json ``` code block.
- Texts must not overlap.
"""

USER_PROMPT = """### Frame Information:
{frame_info}

### Input:
{input_sentence}
"""

frame_descriptions = pd.read_csv('./frame_descriptions_json.csv')
frame_descriptions_mapping = frame_descriptions[['name', 'description']].set_index('name').to_dict()['description']

# Path to base model and LoRA adapter
MAX_TOKENS = 4096  # Maximum tokens for generation (typically don't need more than 4k, but if you change the prompt, you may need to adjust this)
LORA_PATH = "phi-4-fsp-ft-cand"  # Path to LoRA adapter
BASE_MODEL_PATH = "unsloth/phi-4-unsloth-bnb-4bit" # Huggingface or local path


# Initialize vLLM with LoRA adapter
llm = LLM(
    model=BASE_MODEL_PATH,
    enable_lora=True,
    dtype="auto",  # or "float16" for GPU, "bfloat16" if supported
    device="cuda"  # or "cpu" if no GPU
)

fsp_lora = LoRARequest("fsp_lora", 1, LORA_PATH)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-4")

# NOTE: for some reason the LexicalUnitManager needs to be imported after loading vLLM
# Seems to be related to multiprocessing issues with vLLM
from candidate_identifier import LexicalUnitManager
lum = LexicalUnitManager()
lum.load_lus()

def get_candidates(text: str) -> list:
    """
    Get candidate lexical units from the text.

    Args:
        text (str): The input text to analyze.

    Returns:
        list: A list of dictionaries, each containing a target-highlighted string and 
        its corresponding candidate frames.
    """
    candidates = lum.lookup_lus(text)
    
    result = []
    for span, cand_frames in candidates.items():
        highlighted_text = f"{text[:span[0]]}**{text[span[0]:span[1]]}**{text[span[1]:]}"
        result.append({
            "target": highlighted_text,
            "candidates": cand_frames
        })
    
    return result

def parse_json_output(output):
    # Extract XML code from the output
    xml_code = re.search(r'```json(.*?)```', output, re.DOTALL)
    if xml_code:
        cleaned_output = xml_code.group(1).strip()
    
    # Parse the JSON
    try:
        cleaned_output = repair_json(cleaned_output)
        cleaned_output = json.loads(cleaned_output)
    except:
        return {}
    
    return cleaned_output

def frame_semantic_parse(text: str) -> dict:
    """
    Perform frame semantic parsing on the input text using vLLM with LoRA adapter.

    Args:
        text (str): The input text to analyze.
    """

    candidates = get_candidates(text)

    if len(candidates) == 0:
        return pd.DataFrame({
            'target': [text],
            'candidates': [[]],
            'predicted_fes': [{}],
            'original_sentence': [text]
        })
    
    # Create prompts for each candidate
    user_inputs = []
    for candidate in candidates:
        target = candidate['target']
        for frame in candidate['candidates']:
            frame_info = frame_descriptions_mapping[frame]
            prompt = USER_PROMPT.format(frame_info=frame_info, input_sentence=target)
            user_inputs.append({'prompt': prompt, 'frame': frame, 'target': target})
    
    messages = [[{"role": "system", "content": SYSTEM_PROMPT}, 
                {"role": "user", "content": USER_PROMPT['prompt']}] for USER_PROMPT in user_inputs]
    
    messages = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    sampling_params = SamplingParams(
        max_tokens=MAX_TOKENS,
    )

    # Run inference
    responses = llm.generate(
        messages,
        sampling_params=sampling_params,
        lora_request=fsp_lora
    )

    # Parse the responses
    parsed_responses = [parse_json_output(response.outputs[0].text) for response in responses]

    # Combine the results with the original candidates
    user_inputs_df = pd.DataFrame(user_inputs)
    user_inputs_df['predicted_fes'] = parsed_responses
    user_inputs_df['original_sentence'] = text

    return user_inputs_df




# =========== Example usage ===========
text = "The quick brown fox jumps over the lazy dog."
response = frame_semantic_parse(text)

# This code does not remove any empty FE predictions, which may lead to empty JSON objects in the output.
# You may want to filter out empty FE predictions as they are not very useful. (see the commented line below)
# Uncomment the line below to filter out empty predictions
# response = response[response.predicted_fes.apply(lambda x: len(x) > 0)] # Filter out empty predictions

# NOTE: Current implementation does not handle multiple frames for the same target.
# In the paper, we simply randomly selected one as the predicted frame to avoid issues.
# You can also just keep the first prediction for each target if you prefer.
# Uncomment the line below to keep only one prediction per target
# response = response.groupby('target').head(1) # Keep only one prediction per target

response.to_json('response.jsonl', orient='records', lines=True)

