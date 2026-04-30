# Streamlit Cloud 部署说明

1. 将本项目推送到 GitHub 仓库。
2. 登录 Streamlit Cloud，选择 `New app`。
3. Repository 选择该仓库，Branch 选择主分支。
4. Main file path 填写：

```text
app.py
```

5. Streamlit Cloud 会自动读取 `requirements.txt` 安装依赖。
6. 部署完成后打开应用，默认会优先加载 CIFAR-10；如果网络不可用，会自动回退到 scikit-learn digits。

## 注意事项

- 本应用优先使用 torchvision 自动下载 CIFAR-10。若云端网络暂时不可用，会自动使用 scikit-learn digits 真实小数据集，避免页面启动失败。
- PyTorch 在 Streamlit Cloud 上使用 CPU。若响应慢，优先降低训练样本数、训练轮数或每轮训练步数。
- 推荐部署演示参数：图像尺寸 `64x64`，训练样本 `16-32`，epoch `10-50`，每轮步数 `2-4`。
- CIFAR-10 首次下载约 160 MB，云端冷启动可能较慢；应用已提供 digits 回退路径。
