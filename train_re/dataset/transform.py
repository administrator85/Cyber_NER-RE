ID2RELATION_DICT = {"1": "noRelation", "0": "isA", "2": "targets", "3": "uses", "4": "hasAuthor", "5": "variantOf", "6": "hasAlias", "7": "indicates", "8": "exploits"}

# read labels
with open("./labels.txt") as f:
    labels = f.readline()
    labels = labels.split(",")

# read sentences
with open("./sentences.txt") as f:
    sentences = f.readlines()

results = []
categories = {}
import json
for i, sentence in enumerate(sentences):
    relation = ID2RELATION_DICT[labels[i]]
    h = {}
    t = {}
    tokens = []
    import re
    pattern = r'(?<!\d)([.!?;,:])(?!\d)'
    result = re.split(pattern, sentence)
    # 去除空字符串并去掉多余的空格
    result = [s.strip() for s in result if s.strip()]
    # 去掉空格
    for item in result:
        tokens.extend(item.split(" "))
    for j, token in enumerate(tokens):
        if "<e1>" in token:
            h_start = j
            token = token.replace("<e1>", "")
            tokens[j] = token
        if "</e1>" in token:
            h_end = j
            token = token.replace("</e1>", "")
            tokens[j] = token
        if "<e2>" in token:
            t_start = j
            token = token.replace("<e2>", "")
            tokens[j] = token
        if "</e2>" in token:
            t_end = j
            token = token.replace("</e2>", "")
            tokens[j] = token
    h["name"] = " ".join(tokens[h_start : h_end+1])
    h["pos"] = [h_start, h_end+1]
    t["name"] = " ".join(tokens[t_start : t_end+1])
    t["pos"] = [t_start, t_end+1]
    tmp = {"token": tokens, "h": h, "t": t, "relation" : relation}
    json_str = json.dumps(tmp)

    results.append(json_str)
    if relation not in categories:
        categories[relation] = [json_str]
    else:
        categories[relation].append(json_str)

# with open("formed_all.txt", "w") as f:
#     f.write("\n".join(results))
train_list = []
val_list = []
test_list = []
for key, value in categories.items():
    length = len(value)
    train_list.extend(value[:int(length*0.6)])
    val_list.extend(value[int(length*0.6):int(length*0.8)])
    test_list.extend(value[int(length*0.8):])
with open("train.txt", "w") as f:
    f.write("\n".join(train_list))
with open("val.txt", "w") as f:
    f.write("\n".join(val_list))
with open("test.txt", "w") as f:
    f.write("\n".join(test_list))

