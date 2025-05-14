import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import re
import importlib
import yaml
from argparse import Namespace
import os

def load_yaml_config(config_path: str) -> Namespace:
    """从YAML文件加载配置参数"""
    # 校验文件是否存在
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在：{config_path}")
    
    # 读取并解析YAML
    with open(config_path, 'r') as f:
        args_dict = yaml.safe_load(f)
    
    # 转换为Namespace对象
    args = Namespace(**args_dict)
    return args

def _import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'text_recognizer.models.MLP'"""
    print(module_and_class_name)
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    # print(module_name, class_name)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_

def regex_tokenizer(text):
    return re.findall(r"\b\w+(?:['-]\w+)?\b|[,!?;.:]", text)

# 模型加载接口 - 实际上只返回标识符，后续将由本地模型替代
class REModel():
    def __init__(self, model_name = None):
        args = load_yaml_config("re/args.yaml")  # 替换为实际路径
        data_class = _import_class(f"data.{args.data_class}")
        model_class = _import_class(f"models.{args.model_class}")
        litmodel_class = _import_class(f"lit_models.{args.litmodel_class}")
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        model = model_class.from_pretrained(args.model_name_or_path, config=config)
        data = data_class(args, model)
        model.resize_token_embeddings(len(data.tokenizer))
        import torch
        # 初始化模型架构
        lit_model = litmodel_class.load_from_checkpoint(
            checkpoint_path="/data/ner/model/re_model/f1=0.97.ckpt",
            args=args,                # 手动传入必要参数
            model=model,              # 传入预初始化的模型
            tokenizer=data.tokenizer      # 传入tokenizer
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lit_model = lit_model.to(device)
        lit_model.eval()  # 切换到推理模式
        lit_model.freeze()  # 冻结所有参数
        import pytorch_lightning as pl

        # init callbacks
        early_callback = pl.callbacks.EarlyStopping(monitor="Eval/f1", mode="max", patience=5,check_on_train_epoch_end=False)
        model_checkpoint = pl.callbacks.ModelCheckpoint(monitor="Eval/f1", mode="max",
            filename='{epoch}-{Eval/f1:.2f}',
            dirpath="output"
            # save_weights_only=True
        )
        callbacks = [early_callback, model_checkpoint]
        trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, default_root_dir="training/logs")
        self.trainer = trainer
        self.lit_model = lit_model
        self.data = data
        self.args = args

    def create_combinations(self, text, entities):
        tokens = regex_tokenizer(text)
        combinations = []
        for i in range(len(entities)):
            for j in range(len(entities)):
                if j == i:
                    continue
                item = {"token": tokens}
                h_pos = [entities[i]["start"], entities[i]["end"]+1]
                h_name = " ".join(tokens[h_pos[0]:h_pos[1]])
                item["h"] = {"name": h_name, "pos": h_pos}
                t_pos = [entities[j]["start"], entities[j]["end"]+1]
                t_name = " ".join(tokens[t_pos[0]:t_pos[1]])
                item["t"] = {"name": t_name, "pos": t_pos}
                item["relation"] = "noRelation"
                combinations.append(item)
        print(f"Generated {len(combinations)} combinations for relation extraction.")
        import json
        f = open("re/ours/test.txt", "w", encoding="utf-8")
        for item in combinations:
            # 将每个组合写入文件
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        f.close()
        self.combinations = combinations
        

    def extract_relations(self):
        """
        关系抽取接口 - 静态返回结果，后续将由本地模型替代
        
        参数:
            text (str): 原始文本
            entities (list): 实体列表 [{"文本": "张三", "类型": "人名", "开始位置": 0, "结束位置": 2}, ...]
            model_name: 模型标识符
            
        返回:
            relations (list): 关系列表 [{"头实体": "张三", "头实体类型": "人名", "尾实体": "科技公司", "尾实体类型": "机构名", "关系类型": "任职于"}, ...]
        """
        # 这是一个返回静态值的例子
        # 在实际应用中，将在此调用本地RE模型
        self.trainer.test(self.lit_model,datamodule=self.data)
        relations = []
        with open(self.args.results_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped_line = line.strip()
                if stripped_line:
                    try:
                        num = int(stripped_line)
                        relations.append(num)
                    except ValueError:
                        print(f"无法转换行: {line}")
        # translate relation id to relation name
        import json
        RELATION_MAP = json.load(open(self.args.data_dir + "/rel2id.json", "r", encoding="utf-8"))
        REVERSE_MAP = {v: k for k, v in RELATION_MAP.items()}
        for i, relation_id in enumerate(relations):
            relation_name = REVERSE_MAP[relation_id]
            relations[i] = relation_name
        return relations

    def render_relation_graph(self, relations):
        """
        渲染关系图到Streamlit应用
        
        参数:
            relations (list): 关系列表
        """
        if relations:
            # 显示关系表格
            # combanations = []
            # import json
            # with open("/data/ner/app/test/ours/test.txt", "r", encoding="utf-8") as f:
            #     lines = f.readlines()
            #     for line in lines:
            #         combanations.append(json.loads(line.strip()))
            h_entities = [item["h"]["name"] for item in self.combinations]
            t_entities = [item["t"]["name"] for item in self.combinations]
            st.dataframe(pd.DataFrame({"头实体": h_entities, "关系类型": relations, "尾实体": t_entities}))
            
            # 创建关系图
            G = nx.DiGraph()
            
            # 添加节点和边
            for i in range(len(self.combinations)):
                G.add_node(h_entities[i])
                G.add_node(t_entities[i])
                G.add_edge(h_entities[i], t_entities[i], relation=relations[i])
            
            # 绘制图形
            plt.figure(figsize=(10, 8))
            pos = nx.spring_layout(G)
            
            # 绘制节点
            nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=1000)
            
            # 绘制边
            nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=20)
            
            # 绘制标签
            nx.draw_networkx_labels(G, pos, font_size=12)
            
            # 绘制边标签
            edge_labels = {(u, v): d['relation'] for u, v, d in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
            
            # 转换图像为base64以在Streamlit中显示
            buf = BytesIO()
            plt.savefig(buf, format="png", bbox_inches='tight')
            plt.close()
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode()
            st.image(f"data:image/png;base64,{img_str}", caption="实体关系图")
        else:
            st.info("未检测到实体关系")
    def run(self, text, entities):
        """
        执行关系抽取和渲染
        
        参数:
            text (str): 原始文本
            entities (list): 实体列表 [{"文本": "张三", "类型": "人名", "开始位置": 0, "结束位置": 2}, ...]
        
        返回:
            relations (list): 关系列表
        """
        # 创建组合
        self.create_combinations(text, entities)
        
        # 执行关系抽取
        relations = self.extract_relations()
        print(f"******Relations: {relations}")
        # 渲染关系图
        self.render_relation_graph(relations)