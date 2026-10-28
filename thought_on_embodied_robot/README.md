# 具身智能技术路线小书

- **看效果**：按 `Ctrl+Alt+V` 打开内置 PDF 预览，保存 `.tex` 后自动重新编译。
- **对应位置**：在源码中按 `Ctrl+Alt+J` 跳到 PDF；在内置 PDF 中按住 Ctrl 单击可返回源码。
- **改版式**：修改 `bookstyle.tex` 中的字号、页边距、行距和标题设置。
- **改配图**：替换 `assets` 中相应图片，或修改正文中的 `\includegraphics`。
- **命令行编译**：在本目录运行 `python build_pdf.py`。

编译使用 MiKTeX / XeLaTeX。PDF 会更新到根目录，目录、交叉引用和 SyncTeX 等中间文件集中在 `.build`。正文使用原生 LaTeX，Python 只负责调用编译器。


