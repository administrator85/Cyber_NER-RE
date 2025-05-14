import streamlit as st
from re_utils import REModel

# 页面配置
st.set_page_config(
    page_title="关系抽取系统",
    layout="wide"
)

# 设置页面标题
st.title("关系抽取系统")

# 在页面配置后立即加载模型
@st.cache_resource
def init_models():
    # 初始化所有可能用到的模型
    models = {
        "re": {
            "本地RE模型": REModel(),
            "自定义RE模型": None
        }
    }
    return models

# 全局初始化
models = init_models()

# 侧边栏 - 模型选择
with st.sidebar:
    st.header("模型选择")
    
    # RE模型选择
    re_model_option = st.selectbox(
        "选择关系抽取模型", 
        ["本地RE模型", "自定义RE模型"]
    )
        
    if re_model_option == "自定义RE模型" and models["re"]["自定义RE模型"] is None:
        print("Loading custom RE model...")


# 用户输入
st.header("输入文本")
user_input = st.text_area("请输入一段文本进行分析", 
                           "Our research exposes how the malware roots infected devices and steals authentication tokens that can be used to access data from Google Play, Gmail, Google Photos, Google Docs, G Suite, Google Drive, and more.")
uploaded_file = st.file_uploader(
    label="选择文本的命名实体识别结果文件",  # 对话框标题
    type=["json"],  # 允许的文件类型
    accept_multiple_files=False,  # 是否允许多选
    help="请上传 Json 文件"  # 帮助提示
)
if uploaded_file is not None:
    st.success("文件上传成功！")
    # 显示文件基本信息
    st.write("文件名:", uploaded_file.name)
    st.write("文件类型:", uploaded_file.type)
    st.write("文件大小:", uploaded_file.size, "bytes")
col = st.columns(1)[0]

# 处理用户输入
if st.button("开始分析"):
    # 加载所选模型
    try:            
        re_model = models["re"][re_model_option]
        # RE部分
        with col:
            st.header("关系抽取结果")    
            # 处理文本，获取关系
            import json
            entities = json.loads(uploaded_file.read())
            relations = re_model.run(user_input, entities)
    
    except Exception as e:
        st.error(f"发生错误: {e}")

# 添加使用说明
with st.expander("使用说明"):
    st.markdown("""
    ### 使用方法
    1. 在侧边栏选择合适的关系抽取(RE)模型
    2. 在文本框中输入您想要分析的文本
    3. 点击"开始分析"按钮
    4. 查看下方的关系抽取结果
    
    ### 模型说明
    此系统设计为与本地RE模型集成：
    - **RE模型**负责识别实体之间的关系（如任职于、创办、合作等）
    
    ### 集成说明
    要与您自己的模型集成，请修改以下文件：
    - `re_utils.py`：替换`extract_relations`函数中的静态返回值为您的RE模型输出
    
    ### 注意事项
    - 实体识别结果会影响关系抽取的效果
    - 确保您的模型输出格式与系统要求的格式一致
    """)

# 添加开发者信息
st.sidebar.markdown("---")
st.sidebar.info(
    """
    ### 系统信息
    本系统为命名实体识别和关系抽取的基础框架，
    设计用于与本地模型集成。
    """
)