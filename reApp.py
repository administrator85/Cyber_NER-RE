import streamlit as st
from re_utils import REModel
import json
import pandas as pd
import time

# 页面配置
st.set_page_config(
    page_title="关系抽取系统",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "### 关系抽取系统\n专为自然语言处理研究设计"
    }
)

# 设置自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 500;
        color: #0D47A1;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #1E88E5;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #4CAF50;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #FF9800;
        margin-bottom: 1rem;
    }
    .error-box {
        background-color: #FFEBEE;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #F44336;
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
        height: 3rem;
        font-size: 1.2rem;
        font-weight: 500;
        background-color: #1E88E5;
        color: white;
    }
    .stSelectbox>div>div>div {
        background-color: #F5F5F5;
    }
    .stTextArea>div>div>textarea {
        background-color: #F5F5F5;
    }
    .card {
        background-color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
    }
    .relation-tag {
        display: inline-block;
        padding: 0.3rem 0.6rem;
        background-color: #E3F2FD;
        color: #1E88E5;
        border-radius: 15px;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
    }
    .entity-tag {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 3px;
        margin-right: 0.3rem;
        font-weight: 500;
    }
    .entity-person {
        background-color: #E8F5E9;
        color: #388E3C;
    }
    .entity-org {
        background-color: #E3F2FD;
        color: #1976D2;
    }
    .entity-location {
        background-color: #FFF3E0;
        color: #E65100;
    }
    .entity-other {
        background-color: #F5F5F5;
        color: #616161;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #757575;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# 设置页面标题
st.markdown('<p class="main-header">关系抽取系统</p>', unsafe_allow_html=True)

# 在页面配置后立即加载模型
@st.cache_resource
def init_models():
    with st.spinner("加载模型中，请稍候..."):
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

# 边栏设计
with st.sidebar:
    st.markdown('<p class="sub-header">模型控制中心</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">选择适合您任务的模型配置</div>', unsafe_allow_html=True)
    
    # RE模型选择
    re_model_option = st.selectbox(
        "关系抽取模型",
        ["本地RE模型", "自定义RE模型"],
        help="选择用于识别实体间关系的模型"
    )
    
    if re_model_option == "自定义RE模型" and models["re"]["自定义RE模型"] is None:
        st.markdown('<div class="warning-box">⚠️ 自定义模型尚未加载</div>', unsafe_allow_html=True)
    
    # 添加一些模型参数设置
    st.markdown('<p class="sub-header">高级设置</p>', unsafe_allow_html=True)
    confidence_threshold = st.slider(
        "关系置信度阈值", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5,
        help="仅显示置信度高于此阈值的关系"
    )
    
    max_relations = st.slider(
        "最大关系数量", 
        min_value=5, 
        max_value=50, 
        value=20,
        help="最多显示的关系数量"
    )
    
    # 添加开发者信息
    st.markdown("---")
    st.markdown(
        """
        <div class="info-box">
        <b>系统信息</b><br>
        本系统为命名实体识别和关系抽取的高级框架，
        专为研究和实际应用场景设计。
        </div>
        """, 
        unsafe_allow_html=True
    )

# 主界面内容区
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">输入文本</p>', unsafe_allow_html=True)
    
    # 示例文本
    example_texts = {
        "技术文本": "Our research exposes how the malware roots infected devices and steals authentication tokens that can be used to access data from Google Play, Gmail, Google Photos, Google Docs, G Suite, Google Drive, and more.",
        "商业新闻": "Apple Inc. announced that Tim Cook will continue to serve as CEO for the next five years, overseeing operations in Cupertino, California.",
        "自定义文本": ""
    }
    
    selected_example = st.radio("选择示例文本或输入自定义文本:", list(example_texts.keys()))
    
    if selected_example == "自定义文本":
        user_input = st.text_area(
            "请输入一段文本进行分析",
            "",
            height=150,
            help="输入任何您想要分析的文本内容"
        )
    else:
        user_input = st.text_area(
            "文本内容",
            example_texts[selected_example],
            height=150
        )
    
    st.markdown('<p class="sub-header">上传实体识别结果</p>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        label="选择文本的命名实体识别结果文件",
        type=["json"],
        accept_multiple_files=False,
        help="请上传包含实体标注信息的JSON文件"
    )
    
    # 处理用户输入
    analyze_button = st.button(
        "开始分析",
        help="点击开始处理文本并提取关系",
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

# 分析结果展示区
with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">分析结果</p>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        st.markdown('<div class="success-box">✅ 文件上传成功</div>', unsafe_allow_html=True)
        
        with st.expander("文件详情"):
            st.write("文件名:", uploaded_file.name)
            st.write("文件类型:", uploaded_file.type)
            st.write("文件大小:", uploaded_file.size, "bytes")
    else:
        st.markdown('<div class="warning-box">⚠️ 请上传实体识别结果文件</div>', unsafe_allow_html=True)
    
    if analyze_button and user_input and uploaded_file:
        # 显示处理进度
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # 模拟处理过程
            status_text.text("正在加载模型...")
            progress_bar.progress(10)
            time.sleep(0.3)
            
            status_text.text("正在准备数据...")
            progress_bar.progress(30)
            time.sleep(0.3)
            
            # 加载所选模型
            re_model = models["re"][re_model_option]
            
            status_text.text("正在提取关系...")
            progress_bar.progress(60)
            time.sleep(0.3)
            
            # 处理文本，获取关系
            entities = json.loads(uploaded_file.read())
            relations = re_model.run(user_input, entities)
            
            status_text.text("正在整理结果...")
            progress_bar.progress(90)
            time.sleep(0.3)
            
            progress_bar.progress(100)
            status_text.text("分析完成!")
            time.sleep(0.5)
            
            # 清除进度显示
            status_text.empty()
            progress_bar.empty()
            
            # 显示关系抽取结果
            st.markdown('<p class="sub-header">关系抽取结果</p>', unsafe_allow_html=True)
            
            # 将结果转换为DataFrame以便展示
            if relations and len(relations) > 0:
                # 假设relations是包含关系信息的列表
                relation_df = pd.DataFrame(relations)
                
                # 显示关系统计
                st.markdown(f"**共发现 {len(relations)} 个关系**")
                
                # 使用Streamlit的表格组件展示结果
                st.dataframe(relation_df, use_container_width=True)
                
                # 可视化关系（简化版本）
                st.markdown('<p class="sub-header">关系可视化</p>', unsafe_allow_html=True)
                st.markdown("实体与关系的连接图表将在此处显示")
                
                # 这里可以添加一个可视化组件，如使用Streamlit支持的图表库
                # 例如使用NetworkX和Matplotlib创建关系图
                # 由于这部分相对复杂且依赖于具体的数据结构，这里只放置一个占位符
            else:
                st.markdown('<div class="warning-box">未发现任何关系</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.markdown(f'<div class="error-box">⚠️ 发生错误: {e}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# 添加使用说明
with st.expander("使用说明"):
    st.markdown("""
    ### 基本操作流程
    
    1. **选择模型**：在侧边栏选择合适的关系抽取(RE)模型
    2. **输入文本**：在文本框中输入您想要分析的文本，或选择预设示例
    3. **上传实体文件**：上传包含命名实体识别结果的JSON文件
    4. **开始分析**：点击"开始分析"按钮
    5. **查看结果**：在右侧面板查看关系抽取结果和可视化
    
    ### 高级设置说明
    
    - **关系置信度阈值**：调整此滑块可以过滤掉低置信度的关系结果
    - **最大关系数量**：控制最多显示的关系数量，避免信息过载
    
    ### 模型说明
    
    此系统设计为与本地RE模型集成：
    - **RE模型**负责识别实体之间的关系（如任职于、创办、合作等）
    
    ### 集成说明
    
    要与您自己的模型集成，请修改以下文件：
    - `re_utils.py`：替换`extract_relations`函数中的静态返回值为您的RE模型输出
    
    ### 注意事项
    
    - 实体识别结果会影响关系抽取的效果
    - 确保您的模型输出格式与系统要求的格式一致
    - 对于大型文件或复杂文本，处理可能需要更多时间
    """)

# 添加页脚
st.markdown("""
<div class="footer">
    关系抽取系统 • 版本 1.2.0 • © 2025
</div>
""", unsafe_allow_html=True)