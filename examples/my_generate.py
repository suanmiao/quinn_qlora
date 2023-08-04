import os
from os.path import exists, join, isdir
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig
from peft import PeftModel
from peft.tuners.lora import LoraLayer
import argparse

def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, 'completed'))
        if is_completed: return None, True # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        if max_step == 0: return None, is_completed # training started, but no checkpoint
        checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
        print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed # checkpoint found!
    return None, False # first training

# Add command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--max_new_tokens', type=int, default=512)
parser.add_argument('--temperature', type=float, default=0.7)
parser.add_argument('--user_question', type=str, required=True)
parser.add_argument('--model_name_or_path', type=str, required=True)
parser.add_argument('--adapter_path', type=str, required=True)
args = parser.parse_args()

# Update variables
max_new_tokens = args.max_new_tokens
temperature = args.temperature
user_question = args.user_question
model_name_or_path = args.model_name_or_path
adapter_path = args.adapter_path

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
tokenizer.bos_token_id = 1

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    load_in_4bit=True,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
    )
)

model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

prompt = (
    "A chat between a curious human and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions. "
    "### Human: {user_question}"
    "### Assistant: "
)

def generate(model, user_question, max_new_tokens=max_new_tokens, temperature=temperature):
    inputs = tokenizer(prompt.format(user_question=user_question), return_tensors="pt").to('cuda')

    outputs = model.generate(
        **inputs, 
        generation_config=GenerationConfig(
            do_sample=True,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(text)
    return text

generate(model, user_question)