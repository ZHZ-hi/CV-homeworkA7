# Streamlit Cloud 部署说明

1. 将本项目推送到 GitHub 仓库。
2. 登录 Streamlit Cloud，选择 `New app`。
3. Repository 选择该仓库，Branch 选择主分支。
4. Main file path 填写：

```text
app.py
```

5. Streamlit Cloud 会自动读取 `requirements.txt` 安装依赖。
6. 部署完成后打开应用，默认会加载仓库内置的 CIFAR-10 小子集；如果该子集不存在，会尝试下载完整 CIFAR-10，再失败才回退到 scikit-learn digits。

## 注意事项

- 本应用已打包 `data/cifar_subset/`，Streamlit Cloud 无需携带本地 `data/cifar10/` 完整缓存也能展示 CIFAR 图像。
- PyTorch 在 Streamlit Cloud 上使用 CPU。若响应慢，优先降低训练样本数、训练轮数或每轮训练步数。
- 推荐部署演示参数：图像尺寸 `64x64`，训练样本 `16-32`，epoch `10-50`，每轮步数 `2-4`。
- `data/cifar10/` 是本地完整缓存，已在 `.gitignore` 中忽略；不要提交完整 CIFAR-10 压缩包和 batch 文件。
