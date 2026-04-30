import streamlit as st

from modules.data import dataset_source_label, load_sample_images
from modules.train import (
    run_mask_reconstruction_demo,
    run_simclr_demo,
)
from modules.visualize import (
    image_grid,
    metric_table,
    plot_history,
)


st.set_page_config(
    page_title="自监督图像学习演示",
    page_icon="🧠",
    layout="wide",
)


st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    div[data-testid="stMetric"] {
        background: #f5f7f8;
        border: 1px solid #e6eaee;
        border-radius: 8px;
        padding: 12px 14px;
    }
    div[data-testid="stCaptionContainer"] { color: #65717d; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def cached_images(image_size: int, preview_count: int, dataset_marker: str):
    return load_sample_images(image_size=image_size, limit=preview_count)


@st.cache_resource(show_spinner=False)
def cached_mask_demo(
    mask_ratio: float,
    epochs: int,
    seed: int,
    image_size: int,
    sample_count: int,
    learning_rate: float,
    steps_per_epoch: int,
    mask_grid_size: int,
    dataset_marker: str,
):
    images = load_sample_images(image_size=image_size, limit=sample_count)
    return run_mask_reconstruction_demo(
        images,
        mask_ratio=mask_ratio,
        epochs=epochs,
        seed=seed,
        learning_rate=learning_rate,
        steps_per_epoch=steps_per_epoch,
        mask_grid_size=mask_grid_size,
    )


@st.cache_resource(show_spinner=False)
def cached_simclr_demo(
    augment_mode: str,
    epochs: int,
    seed: int,
    image_size: int,
    sample_count: int,
    learning_rate: float,
    steps_per_epoch: int,
    temperature: float,
    top1_margin: float,
    dataset_marker: str,
):
    images = load_sample_images(image_size=image_size, limit=sample_count)
    return run_simclr_demo(
        images,
        augment_mode=augment_mode,
        epochs=epochs,
        seed=seed,
        learning_rate=learning_rate,
        steps_per_epoch=steps_per_epoch,
        temperature=temperature,
        top1_margin=top1_margin,
    )


def format_augment(mode: str) -> str:
    return "随机裁剪 + 颜色扰动" if mode == "crop_color" else "翻转 + 灰度扰动"


def summarize_history(history: list[dict], key: str) -> tuple[float, float, float]:
    start = float(history[0][key])
    end = float(history[-1][key])
    delta = end - start
    return start, end, delta


st.title("图像自监督学习 Web 演示")
st.caption("基于 CIFAR-10 的轻量实验：遮挡补全重建与 SimCLR 对比学习。参数可调，适合课程展示和 Streamlit Cloud 部署。")

dataset_marker = dataset_source_label()

with st.sidebar:
    st.header("实验控制台")
    experiment = st.radio("实验类型", ["遮挡重建", "SimCLR 对比学习"], index=0)

    st.subheader("数据与运行")
    image_size = st.select_slider("图像尺寸", options=[64, 96], value=64)
    preview_count = st.slider("预览样本数", min_value=4, max_value=16, value=8, step=4)
    sample_count = st.slider("训练样本数", min_value=8, max_value=64, value=32, step=8)
    epochs = st.slider("训练轮数", min_value=1, max_value=100, value=50, step=1)
    seed = st.number_input("随机种子", min_value=0, max_value=9999, value=7, step=1)

    if experiment == "遮挡重建":
        st.subheader("遮挡重建参数")
        mask_ratio = st.select_slider("遮挡比例", options=[0.25, 0.50, 0.75], value=0.50, format_func=lambda v: f"{int(v * 100)}%")
        mask_grid_size = st.select_slider("遮挡网格密度", options=[4, 8, 16], value=8, format_func=lambda v: f"{v} x {v}")
        learning_rate = st.select_slider("学习率", options=[0.0005, 0.001, 0.003, 0.005, 0.01], value=0.003, format_func=lambda v: f"{v:g}")
        steps_per_epoch = st.slider("每轮训练步数", min_value=1, max_value=8, value=4, step=1)
        augment_mode = "crop_color"
        temperature = 0.25
        top1_margin = 0.02
    else:
        st.subheader("SimCLR 参数")
        augment_mode = st.selectbox("数据增强方式", ["crop_color", "flip_gray"], format_func=format_augment)
        temperature = st.select_slider("温度系数", options=[0.1, 0.2, 0.25, 0.35, 0.5, 0.8], value=0.25)
        top1_margin = st.select_slider("Top-1 判定间隔", options=[0.0, 0.01, 0.02, 0.05, 0.1], value=0.02)
        learning_rate = st.select_slider("学习率", options=[0.0005, 0.001, 0.003, 0.005, 0.01], value=0.003, format_func=lambda v: f"{v:g}")
        steps_per_epoch = st.slider("每轮增强步数", min_value=1, max_value=8, value=3, step=1)
        mask_ratio = 0.5
        mask_grid_size = 8

    st.subheader("缓存")
    if st.button("清除缓存并重新加载数据", use_container_width=True):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

images = cached_images(image_size, preview_count, dataset_marker)

status_cols = st.columns(4)
status_cols[0].metric("数据源", dataset_marker.replace("真实小数据集：", ""))
status_cols[1].metric("训练样本", f"{sample_count} 张")
status_cols[2].metric("训练轮数", str(epochs))
status_cols[3].metric("图像尺寸", f"{image_size} x {image_size}")

with st.expander("查看数据集样本", expanded=True):
    image_grid([item.image for item in images], [item.name for item in images], columns=4)

st.divider()

if experiment == "遮挡重建":
    st.subheader("遮挡补全重建")
    st.caption(
        f"遮挡比例={int(mask_ratio * 100)}%，遮挡网格={mask_grid_size}x{mask_grid_size}，"
        f"学习率={learning_rate:g}，每轮步数={steps_per_epoch}"
    )

    with st.spinner("正在运行 AutoEncoder 补全训练..."):
        result = cached_mask_demo(
            mask_ratio,
            epochs,
            int(seed),
            image_size,
            sample_count,
            learning_rate,
            steps_per_epoch,
            mask_grid_size,
            dataset_marker,
        )

    before, masked, after, target = result["preview"]
    image_grid(
        [target, masked, before, after],
        ["原始图像", f"遮挡图像（{int(mask_ratio * 100)}%）", "训练前补全区域", f"训练后补全结果（{epochs}轮）"],
        columns=4,
    )

    start_loss, end_loss, delta_loss = summarize_history(result["history"], "loss")
    metric_cols = st.columns(3)
    metric_cols[0].metric("训练前 MSE", f"{start_loss:.4f}")
    metric_cols[1].metric("训练后 MSE", f"{end_loss:.4f}", delta=f"{delta_loss:.4f}")
    metric_cols[2].metric("相对下降", f"{max(0.0, (start_loss - end_loss) / max(start_loss, 1e-9)) * 100:.1f}%")

    chart_col, table_col = st.columns([2, 1])
    with chart_col:
        st.pyplot(plot_history(result["history"], "重建损失 MSE", "loss"))
    with table_col:
        st.markdown("#### 效果对比")
        metric_table(result["comparison"])
        st.info("补全图只替换原来被遮挡的块。遮挡比例越高、网格越粗，任务越难；epoch 和每轮步数越高，通常越接近原图。")

else:
    st.subheader("SimCLR 简化对比学习")
    st.caption(
        f"增强方式={format_augment(augment_mode)}，temperature={temperature}，"
        f"Top-1 margin={top1_margin}，学习率={learning_rate:g}，每轮增强步数={steps_per_epoch}"
    )

    with st.spinner("正在运行轻量 SimCLR 训练..."):
        result = cached_simclr_demo(
            augment_mode,
            epochs,
            int(seed),
            image_size,
            sample_count,
            learning_rate,
            steps_per_epoch,
            temperature,
            top1_margin,
            dataset_marker,
        )

    view_a, view_b = result["views"]
    image_grid([images[0].image, view_a, view_b], ["原始图像", "增强视图 A", "增强视图 B"], columns=3)

    start_loss, end_loss, delta_loss = summarize_history(result["history"], "loss")
    _, end_gap, _ = summarize_history(result["history"], "alignment_gap")
    _, end_top1, _ = summarize_history(result["history"], "top1")
    metric_cols = st.columns(3)
    metric_cols[0].metric("Loss", f"{end_loss:.4f}", delta=f"{delta_loss:.4f}")
    metric_cols[1].metric("正负间隔", f"{end_gap:.4f}")
    metric_cols[2].metric("Top-1", f"{end_top1:.2f}")

    chart_col, table_col = st.columns([2, 1])
    with chart_col:
        st.pyplot(plot_history(result["history"], "对比学习损失与正负样本间隔", "loss", secondary_key="alignment_gap"))
    with table_col:
        st.markdown("#### 效果对比")
        metric_table(result["comparison"])
        st.info("loss 越低越好；正负间隔越大，说明正样本相对负样本更接近；Top-1 表示能否找回同一图像的另一种增强视图。")

st.divider()
with st.expander("部署提示"):
    st.markdown(
        """
        - 默认参数面向 CPU 运行，适合 Streamlit Cloud 演示。
        - 若云端启动较慢，可降低训练样本数、训练轮数或每轮步数。
        - CIFAR-10 会优先自动下载；网络不可用时会回退到 scikit-learn digits。
        - 部署入口文件为 `app.py`，依赖文件为 `requirements.txt`。
        """
    )
