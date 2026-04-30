# 图像自监督学习 Streamlit 演示

这是一个课程作业级 Web 应用，展示两类图像自监督学习任务：

- 遮挡补全重建：随机遮挡输入图像，用轻量 AutoEncoder 重建原图。
- SimCLR 简化对比学习：对同一图像生成两种增强视图，训练编码器拉近正样本表示。

应用优先使用 CIFAR-10 真实小数据集。若运行环境无法下载或尚未缓存 CIFAR-10，会自动使用 scikit-learn 自带的 digits 真实小数据集，最后才退回内置合成样本。

## 本地运行

```powershell
pip install -r requirements.txt
streamlit run app.py
```

## 功能

- 可视化原始图像、遮挡/增强后的图像、模型输出。
- 展示 loss、正负样本间隔、Top-1 匹配率等指标变化。
- 对比训练前后效果。
- 对比不同遮挡比例和不同数据增强设置。
- 支持调节训练样本数、epoch、学习率、每轮训练步数、遮挡网格、SimCLR 温度系数等参数。
- 自动配置 matplotlib 中文字体，避免图像标题乱码。

## 项目结构

```text
app.py
modules/
  data.py
  masking.py
  models.py
  train.py
  visualize.py
data/samples/
requirements.txt
DEPLOYMENT.md
```
