import streamlit as st
from ner_utils import NERModel
from re_utils import REModel

# 页面配置
st.set_page_config(
    page_title="命名实体识别系统",
    layout="wide"
)

# 设置页面标题
st.title("命名实体识别系统")

# 在页面配置后立即加载模型
@st.cache_resource
def init_models():
    # 初始化所有可能用到的模型
    models = {
        "ner": {
            "本地NER模型": NERModel(),
            "自定义NER模型": None  # 按需加载
        },
    }
    return models

# 全局初始化
models = init_models()

# 侧边栏 - 模型选择
with st.sidebar:
    st.header("模型选择")
    
    # NER模型选择
    ner_model_option = st.selectbox(
        "选择NER模型",
        ["本地NER模型", "自定义NER模型"]
    )
    
    # 动态加载未缓存的模型
    if ner_model_option == "自定义NER模型" and models["ner"]["自定义NER模型"] is None:
        print("Loading custom NER model...")



# 用户输入
st.header("输入文本")
user_input = st.text_area("请输入一段文本进行分析", 
                           "Our research exposes how the malware roots infected devices and steals authentication tokens that can be used to access data from Google Play, Gmail, Google Photos, Google Docs, G Suite, Google Drive, and more.")
col = st.columns(1)[0]

# 处理用户输入
if st.button("开始分析"):
    # 加载所选模型
    try:      
        ner_model = models["ner"][ner_model_option]
        # NER部分
        with col:
            st.header("命名实体识别结果")
            # 处理文本，获取实体
            print(user_input)
            entities = ner_model.run(user_input)
            
    except Exception as e:
        st.error(f"发生错误: {e}")

# 添加使用说明
with st.expander("使用说明"):
    st.markdown("""
    ### 使用方法
    1. 在侧边栏选择合适的命名实体识别(NER)模型和关系抽取(RE)模型
    2. 在文本框中输入您想要分析的文本
    3. 点击"开始分析"按钮
    4. 查看左侧的命名实体识别结果和右侧的关系抽取结果
    
    ### 模型说明
    此系统设计为与本地NER模型集成：
    - **NER模型**负责识别文本中的命名实体（如人名、地名、机构名等）
    
    ### 集成说明
    要与您自己的模型集成，请修改以下文件：
    - `ner_utils.py`：替换`process_ner`函数中的静态返回值为您的NER模型输出
    
    ### 注意事项
    - 实体识别结果会影响关系抽取的效果
    - 确保您的模型输出格式与系统要求的格式一致
    """)

# 添加开发者信息
st.sidebar.markdown("---")
st.sidebar.info(
    """
    ### 系统信息
    本系统为命名实体识别的基础框架，
    设计用于与本地模型集成。
    """
)