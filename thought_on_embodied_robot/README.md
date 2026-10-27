# 具身智能技术路线小书

打开 `thought_on_embodied_robot.code-workspace`，在 VS Code 中编辑 `roadmap_for_embodied_tech.tex`。

- **看效果**：按 `Ctrl+Alt+V` 打开内置 PDF 预览，保存 `.tex` 后自动重新编译。
- **对应位置**：在源码中按 `Ctrl+Alt+J` 跳到 PDF；在内置 PDF 中按住 Ctrl 单击可返回源码。
- **改版式**：修改 `bookstyle.tex` 中的字号、页边距、行距和标题设置。
- **改配图**：替换 `assets` 中相应图片，或修改正文中的 `\includegraphics`。
- **命令行编译**：在本目录运行 `python build_pdf.py`。

编译使用你现有的 MiKTeX / XeLaTeX。PDF 会更新到根目录，目录、交叉引用和 SyncTeX 等中间文件集中在 `.build`。正文使用原生 LaTeX，Python 只负责调用编译器。

`roadmap_for_embodied_tech.md` 保留为迁移前的内容底稿。此后的编辑以 `.tex` 为准，日常编译直接使用现有 `.tex`。

## 工作记录

### 2026-10-27

将原有 Markdown 与 Python 绘制的讲义迁移为原生 LaTeX。保留 20 章、3 个附录、23 个公式、6 幅图及 44 条参考资料；补齐项目内图片，配置保存编译、PDF 预览和 SyncTeX。中文使用宋体，英文使用 Times，数学采用 LaTeX 数学字体，表格使用三线表。

完成现有 MiKTeX / XeLaTeX 实编译与全书版面检查：47 页，6 幅配图、44 条参考资料，零溢出版框、零缺字、零未定义引用；已验证 SyncTeX 源码定位。

- 修复 VS Code 编译：当前打开的父目录 `md_for_ML` 增加工作区配置；编译命令明确使用本机 Python 与 MiKTeX 的绝对路径，避免扩展进程的 PATH 影响。独立项目工作区保留相同配置。

- 清理重复配置：当前工作区统一使用父目录 `.vscode/settings.json`；移除子目录的 `settings.json` 和扩展推荐文件，并删除空 `.vscode` 目录。独立 `.code-workspace` 文件仍包含完整编译配置。
