import json
import os

# 将 rootpath 设置为正确的路径
rootpath = "."  # 当前目录

# 列出当前目录下的所有文件
files = os.listdir(rootpath)

# 创建一个用于存储类型信息的字典
types = {"entities": {}, "relations": {}}

for file in files:
    # 只处理 .txt 文件，跳过 readme.txt 和 .json 文件
    if file.endswith('json') or file == "readme.txt" or not file.endswith('.txt'):
        continue

    print(f"Processing file: {file}")  # 打印正在处理的文件名

    rf_path = os.path.join(rootpath, file)
    wf_path = os.path.join(rootpath, file.replace('txt', 'json'))

    # 打开输入文件和输出文件
    with open(rf_path, 'r', encoding='utf-8') as rf, open(wf_path, 'w', encoding='utf-8') as wf:
        datasets = []
        sample = {"tokens": [], "entities": [], "relations": []}
        idx = 0
        doc_id = 0
        start = end = None
        entity_type = None

        for line in rf:
            line = line.strip()
            if line.startswith("-DOCSTART-"):
                doc_id += 1
                continue
            if line:
                last = idx
                fields = list(filter(lambda x: x, line.split()))
                sample["tokens"].append(fields[0])
                sample["orig_id"] = str(doc_id)

                if fields[1].startswith("B-"):
                    if start is not None and end is not None and end == idx:
                        sample["entities"].append({"start": start, "end": end, "type": entity_type})
                    start = idx
                    end = idx + 1  # Initialize end for the new entity
                    entity_type = fields[1][2:]
                    if entity_type not in types["entities"]:
                        types["entities"][entity_type] = {"verbose": entity_type, "short": entity_type}

                if fields[1].startswith("I-"):
                    if end is not None:  # Ensure end is initialized
                        end += 1

                if fields[1] == "O" and start is not None:
                    sample["entities"].append({"start": start, "end": end if end is not None else start + 1, "type": entity_type})
                    start = end = entity_type = None

                idx += 1
            else:
                if start is not None:
                    sample["entities"].append({"start": start, "end": end if end is not None else start + 1, "type": entity_type})
                idx = 0
                start = end = None
                entity_type = None
                if len(sample["tokens"]):
                    datasets.append(sample)
                
                sample = {"tokens": [], "entities": [], "relations": []}

        if len(sample["tokens"]):
            datasets.append(sample)
        
        print(f"Total samples in {file}: {len(datasets)}")
        json.dump(datasets, wf, ensure_ascii=False)

# 保存类型信息到文件
wf_type_path = os.path.join(rootpath, "ner_types.json")
with open(wf_type_path, 'w', encoding='utf-8') as wf_type:
    print(f"Total entity types: {len(types['entities'].keys())}")
    json.dump(types, wf_type, ensure_ascii=False)
