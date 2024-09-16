import os
import sys
import json
import transformers
from typing import Union, List
import fire
import torch
from datasets import load_dataset
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
# model = "meta-llama/Llama-2-7b-chat-hf"

# tokenizer = AutoTokenizer.from_pretrained(model)
# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model,
#     torch_dtype=torch.float16,
#     device_map="auto",
# )

# model = TheModelClass(*args, **kwargs)
# optimizer = TheOptimizerClass(*args, **kwargs)
# checkpoint = torch.load(PATH)
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# epoch = checkpoint['epoch']
# loss = checkpoint['loss']
# model.eval()

# model = PeftModel()

# model = model.push_to_hub(checkpoint_PATH)

# sequences = pipeline(
#     'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
#     do_sample=True,
#     top_k=10,
#     num_return_sequences=1,
#     eos_token_id=tokenizer.eos_token_id,
#     max_length=200,
# )
# for seq in sequences:
#     print(f"Result: {seq['generated_text']}")

def set_environment_variables():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    os.environ["WANDB_DISABLED"] = "true"

def check_cuda_availability():
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print("Number of GPUs:", device_count)
        for i in range(device_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available.")


def load_pretrained_model(base_model: str, device_map: str):
    config = BitsAndBytesConfig(load_in_8bit=True)
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        quantization_config=config,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    model = prepare_model_for_kbit_training(model)
    return model

def load_tokenizer(base_model: str):
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token
    tokenizer.padding_side = "left"
    return tokenizer

def load_data(data_path: str):
    data_files = {
        'train': data_path,
        'test': data_path.replace("train", "dev")
    }
    data = load_dataset('json', data_files=data_files)
    return data

def generate_prompt(data_point, template):
    # 这里一定要对应好 json 数据中的 key.
    instruction = data_point['instruction']
    input_text = data_point.get('input', None)
    label = data_point.get('output', None)
    
    if input_text:
        res = template["prompt_input"].format(instruction=instruction, input=input_text)
    else:
        res = template["prompt_no_input"].format(instruction=instruction)
    if label:
        res = f"{res}{label}"
    
    return res

def tokenize(prompt, tokenizer, add_eos_token=True, cutoff_len=256):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=True,  
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()
    return result

def generate_and_tokenize_prompt(data_point, template, tokenizer, add_eos_token, cutoff_len):
    full_prompt = generate_prompt(data_point, template)
    tokenized_full_prompt = tokenize(full_prompt, tokenizer, add_eos_token, cutoff_len)
    return tokenized_full_prompt

def split_train_val(data, val_set_size, seed=42):
    train_val = data["train"].train_test_split(test_size=val_set_size, shuffle=True, seed=seed)
    return train_val

def configure_lora(lora_r, lora_alpha, lora_dropout, lora_target_modules):
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return lora_config

def train_model(
     # model/data params
    base_model: str = "/home/1013lwb/.cache/modelscope/hub/skyline2006/llama-7b",  # the only required argument
    data_path: str = "/home/1013lwb/work/work/RTE/train.json",
    output_dir: str = "/home/1013lwb/work/work/output",
    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: int = 2000,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_path: str = "/home/1013lwb/work/work/RTE/template.json",  # The prompt template to use, will default to alpaca.
):
    set_environment_variables()
    check_cuda_availability()

    model = load_pretrained_model(base_model, "auto")
    tokenizer = load_tokenizer(base_model)

    checkpoint_path = '/home/1013lwb/work/work/output_rte/checkpoint-50'

    model = PeftModel.from_pretrained(model, model_id=checkpoint_path)
    model = model.to("cuda")
    model.eval()
    # inputs = tokenizer("Preheat the oven to 350 degrees and place the cookie dough", return_tensors="pt")

    # outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=50)
    # print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0])

    data = load_data(data_path)
    
    with open(prompt_template_path, 'r') as file:
        template = json.load(file)

    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )

    train_val = split_train_val(data, val_set_size)
    train_data = train_val["train"].map(lambda x: generate_and_tokenize_prompt(x, template, tokenizer, add_eos_token, cutoff_len))
    val_data = train_val["test"].map(lambda x: generate_and_tokenize_prompt(x, template, tokenizer, add_eos_token, cutoff_len))
    
    # lora_config = configure_lora(lora_r, lora_alpha, lora_dropout, lora_target_modules)

    # gradient_accumulation_steps = batch_size // micro_batch_size
   
    gradient_accumulation_steps = batch_size // micro_batch_size

    output_file="/home/1013lwb/work/work/output.txt"

    with open(output_file, 'w') as f:
        for item in train_data:
            #print(item['input_ids'])
            tensor = torch.LongTensor(item['input_ids'])
            attention_mask = torch.Tensor(item['attention_mask'])
            #print(tensor)
            outputs = model.generate(
                tensor.to("cuda"), 
                attention_mask=attention_mask,
                eos_token_id=29900, 
                max_new_tokens=50
            )
            print(outputs, file=f)


if __name__ == "__main__":
    train_model()
