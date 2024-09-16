import os
import sys
import re
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

def set_environment_variables():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1" #注意：如果改为0,1使用两个GPU的话会报错，解决方案不明
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
        #quantization_config=config,   
        #注意：量化后如果不载入我们自己训练的checkpoint会报错，内容为8bit模型无法使用.to命令，载入checkpoint则不会报错
        #怀疑是载入那一步会取消掉8bit的设置，所以我直接注释掉了
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

def extract_bracket_contents(input_string):
    # 使用正则表达式匹配大括号中的内容
    pattern = r'\{(.*?)\}'
    matches = re.findall(pattern, input_string)
    return matches

def generate_prompt(data_point, template):
    # 旧版注释：这里一定要对应好 json 数据中的 key.
    # instruction = data_point['instruction']
    # input_text = data_point.get('input', None)
    # label = data_point.get('output', None)
    
    # if input_text:
    #     res = template["prompt_input"].format(instruction=instruction, input=input_text)
    # else:
    #     res = template["prompt_no_input"].format(instruction=instruction)
    # if label:
    #     res = f"{res}{label}"
    
    # return res

    #新版注释
    #想统一处理，数据集的不同由template.json来控制,提取template中的大括号中和其中内容，然后将data_point中查找到的相关内容填充进去
    # Generate a prompt based on the given data point and template.
    # Args:
    #     data_point (dict): The data point containing the necessary information.
    #     template (dict): The template containing the prompt structure.
    # Returns:
    #     str: The generated prompt.
    # Raises:
    #     KeyError: If a key in the template is not found in the data point.

    bracket_contents = extract_bracket_contents(template["prompt_input"])
    # 提取大括号中的内容，理想状态下包括Response, Conclusion, Reason等
    #print("bracket_contents:", bracket_contents)
    fill_contents = {}
    for item in bracket_contents:
        try:
            #fill_contents.append(data_point.get(item, None))
            fill_contents[item] = data_point.get(item, None)
        except:
            print(f"filled item not found in data_point")
    #print("fill_contents", fill_contents)
    res = template["prompt_input"]
    res = res.format(**fill_contents)
    return res

def tokenize(prompt, tokenizer, add_eos_token=True, cutoff_len=256):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding="max_length",
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

def split_train_val(data, val_set_size, data_catagory,seed=42):
    train_val = data[data_catagory].train_test_split(test_size=val_set_size, shuffle=True, seed=seed)
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
def calculate_accuracy(result, train_data,task_type):
    # result is a list of strings, each string is a response from the model
    # train_data is a list of dictionaries, each dictionary is a data point

    total = len(result)
    if task_type == "RTE" or task_type == "CoLA" or task_type == "SST2" or task_type == "MRPC" or task_type == "QQP" or task_type == "QNLI" or task_type == "WNLI":
        TP = 0 
        TN = 0
        FP = 0
        FN = 0
        for i,result_point in enumerate(result):
            answer_str = train_data[i]['output'] 
            answer = float(answer_str)
            try:
                Response_locate = result_point.index("Response:")
            except:
                print("Response not found in origin_index:", train_data[i]["index"])
                total -= 1
                continue
            try:
                Conclusion_locate = result_point.index("Conclusion")
            except:
                Conclusion_locate = -1

            result_point = result_point.lower()
            tmp1 = result_point.find(' not ', Response_locate,Conclusion_locate)
            tmp2 = result_point.find(' no ', Response_locate,Conclusion_locate)
            tmp3 = result_point.find('false', Response_locate,Conclusion_locate)
            tmp4 = result_point.find('negative', Response_locate,Conclusion_locate)
            tmp = max(tmp1, tmp2, tmp3,tmp4)
            if tmp == -1:
                point_is_pos = 1
            else:
                point_is_pos = -1
            # print(f"point_is_pos =", point_is_pos, " answer =", answer)
            if point_is_pos == answer:
                if answer == 1:
                    TP += 1
                else:
                    TN += 1
            else:
                if answer == 1:
                    FN += 1
                else:
                    FP += 1
            # print(f"right_ans = ", right_ans)
        precision = -1
        recall = -1
        f1 = -1
        acc = (TP + TN) / total
        if TP + FP != 0:
            precision = TP / (TP + FP)
        if TP + FN != 0:
            recall = TP / (TP + FN)
        if precision + recall != 0:
            f1 = 2 * precision * recall / (precision + recall)

        print(f"TP = {TP} TN = {TN} FP = {FP} FN = {FN}")
        print(f"total = {total}")
        print(f"acc = {acc} precision = {precision} recall = {recall} f1 = {f1}")
    
    if task_type == "STSB":
        right_ans = 0
        wrong_ans = 0
        for i,result_point in enumerate(result):
            answer_str = train_data[i]['output'] 
            answer = float(answer_str)
            try:
                Response_locate = result_point.index("Response:")
            except:
                print("Response not found in origin_index:", train_data[i]["index"])
                total -= 1
                continue
            try:
                Conclusion_locate = result_point.index("Conclusion")
            except:
                Conclusion_locate = -1

            try:
                tmp = re.findall(r"\d+\.?\d*", result_point[Response_locate:Conclusion_locate])
            except:
                print("No number found in origin_index:", train_data[i]["index"])
                total -= 1
                continue            
            
            f = False #标记是否找到合适的数字 
            
            for i,item in enumerate(tmp):
                item = float(item)
                if item >= 0 and item <= 5:
                    point_num = item
                    f = True
                    break
            if f == False:
                print("No proper number found in origin_index:", train_data[i]["index"])
                total -= 1
                continue

            # print(f"point_is_pos =", point_is_pos, " answer =", answer)
            if point_num - answer < 0.5: #如果预测的数值和标签的数值差距小于0.5，则认为预测正确
                right_ans += 1
            else:
                wrong_ans += 1
            # print(f"right_ans = ", right_ans)
            
        acc = right_ans / total
        print(f"total = {total}")
        print(f"right_ans = {right_ans} wrong_ans = {wrong_ans}")
        print(f"acc = {acc}")

    if task_type == "MNLI":
        right_ans = 0
        wrong_ans = 0
        for i,result_point in enumerate(result):
            answer_str = train_data[i]['output'] 
            answer_str =answer_str.lower()
            
            try:
                Response_locate = result_point.index("Response:")
            except:
                print("Response not found in origin_index:", train_data[i]["index"])
                total -= 1
                continue
            try:
                Conclusion_locate = result_point.index("Conclusion")
            except:
                Conclusion_locate = -1

            result_point = result_point.lower()

            tmp1 = result_point.find('contradiction', Response_locate,Conclusion_locate)
            tmp2 = result_point.find('neutral', Response_locate,Conclusion_locate)
            tmp3 = result_point.find('entailment', Response_locate,Conclusion_locate)

            num_tmp_over0 = 0
            if tmp1 >= 0:
                num_tmp_over0 += 1
            if tmp2 >= 0:
                num_tmp_over0 += 1
            if tmp3 >= 0:
                num_tmp_over0 += 1
            if num_tmp_over0 >= 2 or num_tmp_over0 == 0:
                print("Find over one answer in origin_index:", train_data[i]["index"])
                total -= 1
                continue

            if(answer_str == 'contradiction' and tmp1 >= 0) or (answer_str == 'neutral' and tmp2 >= 0) or (answer_str == 'entailment' and tmp3 >= 0):
                right_ans+=1
            else:
                wrong_ans+=1

        acc = right_ans / total
        print(f"total = {total}")
        print(f"right_ans = {right_ans} wrong_ans = {wrong_ans}")
        print(f"acc = {acc}")
    return acc

def generate(
     # model/data params
    base_model: str = "/home/1013lwb/.cache/modelscope/hub/skyline2006/llama-7b",  # 注意：不要更改的大模型路径
    data_path: str = "../glue/RTE/train.json", #注意：需要更改的数据集路径
    data_catagory: str = "train",  # options: train | test(dev) 注意：前者为选取train.json后者为选取dev.json 
    generate_train_or_test: str = "train",  # options: train | test(dev) 注意：这里选的是split_train_val中的train or test
    output_file: str = "../output/output_for_RTE.txt",
    task_type: str = "RTE",  
    # training hyperparams
    micro_batch_size: int = 4, #必须为2次幂，因为后面用到&运算
    cutoff_len: int = 256,
    val_set_size: int = 2000, # float or int ,代表用于验证的数据集大小，注意是split的参数
    # llm hyperparams
    add_eos_token: bool = False,
    # wandb params
    prompt_template_path: str = "../glue/RTE/template_RTE.json",  # The prompt template to use, will default to alpaca.
):
    set_environment_variables()
    check_cuda_availability()

    model = load_pretrained_model(base_model, "auto")
    tokenizer = load_tokenizer(base_model)

    # checkpoint_path = '/home/1013lwb/work/work/output_RTE'

    # model = PeftModel.from_pretrained(model, model_id=checkpoint_path)

    # model = torch.nn.DataParallel(model)
    model = model.to("cuda")
    model.eval()

    data = load_data(data_path)
    with open(prompt_template_path, 'r') as file:
        template = json.load(file)

    train_val = split_train_val(data = data, data_catagory = data_catagory ,val_set_size = val_set_size,) 
    train_data = train_val[generate_train_or_test].map(lambda x: generate_and_tokenize_prompt(x, template, tokenizer, add_eos_token, cutoff_len))    

    result = []
    micro_batch = 0
    tot_train_data = len(train_data)
    tot_batch = tot_train_data // micro_batch_size

    print("tot_batch= ", tot_batch)
    
    for i,item in enumerate(train_data):
        if i & (micro_batch_size - 1) == 0: #每个micro_batch开始，创建一个新的data_for_generate
            data_for_generate = {
                'input_ids': [],
                'attention_mask': []
            } 
        data_for_generate['input_ids'].append(item["input_ids"])
        data_for_generate['attention_mask'].append(item["attention_mask"])
        # if i == 16:          #测试小样本用
        #     break
        if i & (micro_batch_size - 1) == (micro_batch_size - 1):
            micro_batch += 1
            print("micro_batch = ", micro_batch, "/", tot_batch) #当前跑到第几个micro_batch

            data_for_generate['input_ids'] = torch.tensor(data_for_generate['input_ids']).to("cuda")
            data_for_generate['attention_mask'] = torch.tensor(data_for_generate['attention_mask']).to("cuda")
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=data_for_generate['input_ids'], 
                    attention_mask=data_for_generate['attention_mask'],
                    eos_token_id=tokenizer.eos_token_id, 
                    max_new_tokens=100
                )
            with open(output_file, 'a') as f:
                for j in range(len(outputs)):
                    decoded_output = tokenizer.decode(outputs[j])
                    print("No.",i-micro_batch_size+j+1,file=f)
                    try:
                        print("original index",train_data[i-micro_batch_size+j+1]["index"],file=f)
                    except:
                        print("original index not found",file=f)
                    print(decoded_output+'\n', file=f)
                    result.append(decoded_output)
    acc = calculate_accuracy(result, train_data,task_type)
    
if __name__ == "__main__":
    generate()