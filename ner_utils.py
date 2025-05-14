import streamlit as st
import re
from prompt4ner.prompt4ner_trainer import Prompt4NERTrainer
from prompt4ner import input_reader

def regex_tokenizer(text):
    return re.findall(r"\b\w+(?:['-]\w+)?\b|[,!?;.:]", text)
class NERModel:
    def __init__(self):
        model_path = "/data/ner/model/ner_finetuned"
        self.model = Prompt4NERTrainer(model_path)
    
    def inference_ner(self, text):
        """
        NER处理接口 - 静态返回结果，后续将由本地模型替代
        
        参数:
            text (str): 要处理的文本
            model_name: 模型标识符
            
        返回:
            entities (list): 实体列表 [{"文本": "张三", "类型": "PERSON", "开始位置": 0, "结束位置": 2}, ...]
        """
        # 在实际应用中，将在此调用本地NER模型
        # 处理输入文本
        tokens = regex_tokenizer(text)
        data = [{"tokens": tokens,"entities": [],"relations": [],"orig_id": "1"}]
        import json
        tmp_path = "ner/input.json"
        json.dump(data, open(tmp_path, "w", encoding="utf-8"), ensure_ascii=False)    
        # 调用模型
        self.model.eval(tmp_path, "ner/ner_types.json", input_reader_cls=input_reader.JsonInputReader)
        json_output = self.model.result
        entities = json_output[0]["entities"]
        print(entities)
        return entities

    def render_ner_results(self, text, entities):
        """
        渲染NER结果到Streamlit应用
        
        参数:
            text (str): 原始文本
            entities (list): 实体列表 [{"文本": "张三", "类型": "人名", "开始位置": 0, "结束位置": 2}, ...]
        """
        if entities:
            # 显示实体表格
            # st.dataframe(pd.DataFrame(entities))
            
            # 创建高亮显示文本
            # 将文本拆分为单词（根据需求可能用于其他处理）
            tokens = regex_tokenizer(text)

            # 定义实体类型对应的颜色映射
            color_map = {
                "Application": "#FF9999",
                # 添加其他实体类型和颜色
            }

            # 按实体起始位置排序
            sorted_entities = sorted(entities, key=lambda x: x["start"])

            last_end = 0
            html_content = ["<div>"]  # 使用列表提高字符串拼接效率

            for ent in sorted_entities:
                # 处理文本区间重叠的情况，仅处理未覆盖部分
                effective_start = max(ent["start"], last_end)
                effective_end = ent["end"]
                
                # 跳过无效区间或完全重叠的实体
                if effective_start > effective_end:
                    continue
                
                # 添加实体前的普通文本
                if effective_start > last_end:
                    plain_text = tokens[last_end:effective_start]
                    plain_text = " ".join(plain_text)
                    html_content.append(plain_text)
                
                # 处理实体部分
                if effective_start <= effective_end:
                    entity_text = tokens[effective_start:effective_end + 1]
                    entity_text = " ".join(entity_text)
                    entity_type = ent['entity_type']
                    color = color_map.get(ent["entity_type"], "#DDDDDD")
                    
                    # 构建高亮标记
                    highlight = (
                        f"<mark style='background: {color}; padding: 0.1em 0.3em; border-radius: 0.35em;'>"
                        f"{entity_text}"
                        f"<span style='font-size: 0.8em; font-weight: bold; margin-left: 0.5rem;'>{entity_type}</span>"
                        "</mark>"
                    )
                    html_content.append(highlight)
                
                # 更新区间结束位置（end为inclusive故+1）
                last_end = max(last_end, effective_end + 1)

            # 添加末尾剩余文本
            if last_end < len(tokens):
                trailing_text = tokens[last_end:]
                trailing_text = " ".join(trailing_text)
                html_content.append(trailing_text)

            html_content.append("</div>")
            html = "".join(html_content)
            
            st.write(html, unsafe_allow_html=True)
        else:
            st.info("未检测到命名实体")

    def run(self, text):
        # 进行NER处理
        entities = self.inference_ner(text)
        import json
        json.dump(entities, open("ner/ner.json", "w", encoding="utf-8"), ensure_ascii=False)

        with open("ner/ner.json", "rb") as f:
            st.download_button(
                label="ner结果下载",
                data=f,
                file_name="ner.json",
                mime="text/json",
            )
        # 渲染NER结果
        self.render_ner_results(text, entities)
        return entities    