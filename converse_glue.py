import csv
import json
import os

TASKS = ["CoLA", "SST", "MRPC", "QQP", "STS", "MNLI", "QNLI", "RTE", "WNLI", "diagnostic"]


def tsc_to_one_line_json_for_RTE(tsc_file_path=None, json_file_path=None):
    base_file_path = 'F:/laboratory/work/CoLA'  # 替换为你的 TSC 文件路径
    file_name = ['train', 'test', 'dev']

    for name in file_name:
        tsc_file_path = base_file_path + f"/{name}.tsv"
        json_file_path = base_file_path + f"/{name}.json"

        try:
            # 获取当前工作目录并打印
            cwd = os.getcwd()
            print(f"Current working directory: {cwd}")

            # 检查文件是否存在
            if not os.path.isfile(tsc_file_path):
                print(f"TSC file does not exist: {tsc_file_path}")
                return
            
            # 读取 TSC 文件
            with open(tsc_file_path, 'r', encoding='utf-8') as tsc_file:
                reader = csv.DictReader(tsc_file, delimiter='\t')
                
                # 将 TSC 数据转换为字典列表
                data = []
                for row in reader:
                    index = row['index']
                    sentence1 = row['sentence1']
                    sentence2 = row['sentence2']
                    label = row['label']
                    if name != "test":
                        # need to be str in json.
                        label = "1" if row['label'] == 'entailment' else "0"
                        data.append({
                            'index': index,
                            'instruction': sentence1,
                            'input': sentence2,
                            'output': label
                        })
                    else:
                        data.append({
                            'index': index,
                            'instruction': sentence1,
                            'output': sentence2,
                        })
            
            # 保存为 JSON 文件 (一行一个对象)
            with open(json_file_path, 'w', encoding='utf-8') as json_file:
                for item in data:
                    # ensure_ascii=False 用来保证部分字符可以正确显示
                    json_file.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"Successfully converted {tsc_file_path} to {json_file_path}")
        
        except Exception as e:
            print(f"Error: {e}")


# 示例调用
if __name__ == "__main__":
    tsc_to_one_line_json_for_RTE()