### 原型系统（APP）项目说明文档

---

#### 一、源代码总体说明

**1. 项目目录结构**  

```
app/  
├── data/                 # re模型中数据处理代码  
├── lit_models/           # re模型中PyTorch Lightning模型代码  
├── models/               # re模型中模型代码  
├── ner/                  # NER模型运行参数、数据和结果 
├── prompt4ner/           # ner模型 
├── re/                   # re模型运行参数、数据和结果 
├── ner_utils.py          # NER 页面前端代码  
├── nerApp.py             # NER APP 入口
├── re_utils.py           # RE 页面前端代码 
└── reApp.py              # RE APP 入口  
```

---

#### 二、运行环境配置

**1. 基础环境**  

```yaml
OS: Ubuntu 20.04 LTS  
GPU: NVIDIA 4090 * 2 (24GB VRAM * 2)  
CUDA: 12.1  
cuDNN: 8.9.0  
Python: 3.8.12  
```

**2. 依赖安装**  

```bash
# 创建ner环境
conda create -n ner python=3.8
conda activate ner
# 通过requirements.txt一键安装  
pip install -r requirements.txt  
```

```ini
# requirements.txt 内容  
streamlit
torch
transformers==4.20.1    
tqdm==4.54.0
Jinja2==3.0.1
scikit-learn==0.23.2
numpy==1.19.2
scipy==1.5.4
pynvml==8.0.4
tensorboard==2.13.0
pillow==8.1.2
```

```bash
# 创建re环境
conda create -n re python=3.9
conda activate re
# 通过requirements.txt一键安装  
pip install -r requirements.txt  
```

```ini
# requirements.txt 内容  
streamlit
numpy==1.20.3
tokenizers==0.10.3
pytorch_lightning==1.3.1
regex==2024.11.6
torch==1.13.1
transformers==4.7.0
tqdm==4.49.0
activations==0.1.0
dataclasses==0.6
file_utils==0.0.1
flax==0.3.4
PyYAML==5.4.1
utils==1.0.1
```

---

#### 三、执行流程说明

**0. 环境激活**  

```bash  
conda activate re
conda activate ner
```

**1. 运行脚本**  

```bash  
streamlit run re
streamlit run ner
```

**2. 打开网页**  
