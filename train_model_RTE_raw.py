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
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig

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

def configure_training_arguments(output_dir, micro_batch_size, gradient_accumulation_steps, num_epochs, learning_rate):
    training_arguments = transformers.TrainingArguments(
         per_device_train_batch_size=micro_batch_size,
         gradient_accumulation_steps=gradient_accumulation_steps,
         warmup_steps=100,
         max_steps=num_epochs * gradient_accumulation_steps,
         learning_rate=learning_rate,
         fp16=True,
         logging_steps=10,
         optim="adamw_torch",
         evaluation_strategy="steps",
         save_strategy="steps",
         eval_steps=50,
         save_steps=50,
         output_dir=output_dir,
         save_total_limit=3,
         load_best_model_at_end=True,
         report_to="none"
    )
    return training_arguments

def train_model(
     # model/data params
    base_model: str = "/home/1013lwb/.cache/modelscope/hub/skyline2006/llama-7b/",  # the only required argument
    data_path: str = "./RTE/train.json",
    output_dir: str = "./output_raw",
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
    prompt_template_path: str = "./RTE/template.json",  # The prompt template to use, will default to alpaca.
):
    set_environment_variables()
    check_cuda_availability()

    model = load_pretrained_model(base_model, "auto")
    tokenizer = load_tokenizer(base_model)
    
    data = load_data(data_path)
    
    with open(prompt_template_path, 'r') as file:
        template = json.load(file)

    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )

    train_val = split_train_val(data, val_set_size)
    train_data = train_val["train"].map(lambda x: generate_and_tokenize_prompt(x, template, tokenizer, add_eos_token, cutoff_len))
    val_data = train_val["test"].map(lambda x: generate_and_tokenize_prompt(x, template, tokenizer, add_eos_token, cutoff_len))
    
    lora_config = configure_lora(lora_r, lora_alpha, lora_dropout, lora_target_modules)
    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    gradient_accumulation_steps = batch_size // micro_batch_size
    training_arguments = configure_training_arguments(output_dir, micro_batch_size, gradient_accumulation_steps, num_epochs, learning_rate)
    
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_arguments,
        data_collator=data_collator
    )
    
    model.config.use_cache = False
    
    # 这些行需要被注释，否则在模型训练最后会报错
    # old_state_dict = model.state_dict
    # model.state_dict = (
    #     lambda self, *_, **__: get_peft_model_state_dict(
    #         self, old_state_dict()
    #     )
    # ).__get__(model, type(model))
    # model = torch.compile(model)
    
    trainer.train()

    model.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(train_model)
