# 自动化之后--具身智能学习笔记

**中文** · [English](README.en.md)

从深度学习到强化学习与机器人任务，记录概念、公式推导和学习过程。

其中**Markdown 用于阅读源码、Fork 和共同完善；PDF 用于完整阅读、下载与打印。**

## 课程资料

点击表中链接阅读或下载。

| 序号 | 主题 | Markdown | PDF | 内容范围 |
| :---: | --- | :---: | :---: | --- |
| 1 | KNN：K 近邻方法 | [MD](deep_learning/KNN.md) | [PDF](https://acsg999.github.io/advanced-automation-notes/deep_learning/KNN.pdf) | 基于实例的学习、分类与回归、邻居数量与距离 |
| 2 | MLP：多层感知机 | [MD](deep_learning/mlp.md) | [PDF](https://acsg999.github.io/advanced-automation-notes/deep_learning/mlp.pdf) | 前向传播、反向传播、梯度更新与动量 |
| 3 | 西湖大学强化学习 | [MD](rl-xihu/西湖大学强化学习.md) | [PDF](https://acsg999.github.io/advanced-automation-notes/rl-xihu/%E8%A5%BF%E6%B9%96%E5%A4%A7%E5%AD%A6%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0.pdf) | Bellman 方程、价值迭代、MC、TD、DQN、策略梯度与 Actor-Critic |
| 4 | 具身智能技术路线小书 | [latex](thought_on_embodied_robot/roadmap_for_embodied_tech.tex) | [PDF](https://acsg999.github.io/advanced-automation-notes/thought_on_embodied_robot/roadmap_for_embodied_tech.pdf) | 任务建模、VLA、力触觉、世界模型、系统组合与评测 |


## 如何阅读

- **学习**：下载 PDF，按目录阅读；原稿中的公式、图片和补充解释保留在讲义中。
- **查阅与修改**：打开 MD，利用 GitHub 的源码、差异对比和历史记录追踪修改。
- **具身智能讲义排版**：查看[编辑与编译说明](thought_on_embodied_robot/README.md)。

## 共同完善

欢迎fork本仓库，补充概念解释、纠正公式、改善配图与引用，也欢迎提交学习中遇到的问题。

现有笔记侧重学习记录。发现推导或概念问题时，可在 [Issues](https://github.com/AcSg999/advanced-automation-notes/issues) 中标出主题和具体位置。

## 重建 PDF

需要 **Python 3.9+、Pandoc 3.9+ 和 XeLaTeX**（TeX Live 或 MiKTeX），以及宋体、黑体、Times New Roman、Consolas 字体。Python 构建脚本只使用标准库。

在仓库根目录运行：

```bash
# 补齐缺失的 PDF；已有 PDF 保留
python scripts/build_pdfs.py

# 修改正文后，指定需要重建的主题：knn / mlp / rl / embodied
python scripts/build_pdfs.py mlp

# 重建全部 4 份 PDF
python scripts/build_pdfs.py --all
```



