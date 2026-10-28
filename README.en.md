# Beyond Automation: Notes on Embodied Intelligence

[中文](README.md) · **English**

From deep learning to reinforcement learning and robot tasks, these notes document concepts, mathematical derivations, and the learning process.

**Markdown supports source reading, forks, and collaborative editing. PDFs provide typeset editions for reading, downloading, and printing.**

## Course materials

Use the links below to read or download the materials.

| No. | Topic | Markdown | PDF | Coverage |
| :---: | --- | :---: | :---: | --- |
| 1 | K-Nearest Neighbors | [MD](deep_learning/KNN.md) | [PDF](deep_learning/KNN.pdf) | Instance-based learning, classification and regression, neighbor count, and distance |
| 2 | Multilayer Perceptrons | [MD](deep_learning/mlp.md) | [PDF](deep_learning/mlp.pdf) | Forward propagation, backpropagation, gradient updates, and momentum |
| 3 | Westlake University Reinforcement Learning Notes | [MD](rl-xihu/西湖大学强化学习.md) | [PDF](rl-xihu/西湖大学强化学习.pdf) | Bellman equations, value iteration, MC, TD, DQN, policy gradients, and Actor-Critic |
| 4 | A Short Guide to Technical Routes in Embodied Intelligence | [LaTeX](thought_on_embodied_robot/roadmap_for_embodied_tech.tex) | [PDF](thought_on_embodied_robot/roadmap_for_embodied_tech.pdf) | Task modeling, VLA, force and touch, world models, system integration, and evaluation |

## Reading

- **Study**: download a PDF and follow its contents. Equations, images, and explanatory notes are retained from the source.
- **Review and editing**: open the MD and use GitHub's source view, diffs, and history to follow changes.
- **Editing the embodied intelligence guide**: see its [editing and build instructions](thought_on_embodied_robot/README.md) in Chinese.

## Contributing

Feel free to fork this repository, clarify concepts, correct equations, improve figures and references, or share questions encountered while studying.

These are learning notes. To report a conceptual or derivation issue, identify the topic and location in [Issues](https://github.com/AcSg999/advanced-automation-notes/issues).

## Rebuilding PDFs

Install **Python 3.9+, Pandoc 3.9+, and XeLaTeX** (TeX Live or MiKTeX), with SimSun, SimHei, Times New Roman, and Consolas fonts. The Python build script uses only the standard library.

Run from the repository root:

```bash
# Generate missing PDFs, preserving existing ones
python scripts/build_pdfs.py

# Rebuild a topic after editing: knn / mlp / rl / embodied
python scripts/build_pdfs.py mlp

# Rebuild all 4 PDFs
python scripts/build_pdfs.py --all
```

