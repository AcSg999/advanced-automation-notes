"""读取指定的 Markdown 文件，在原目录生成同名讲义 PDF。"""

from pathlib import Path
import argparse
import json
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory


SCRIPTS = Path(__file__).resolve().parent


def executable(name, fallback=None):
    found = shutil.which(name)
    if found:
        return found
    if fallback and Path(fallback).is_file():
        return fallback
    raise RuntimeError(f"找不到 {name}，请先安装并加入 PATH。")


def prepare_markdown(source):
    text = source.read_text(encoding="utf-8")
    # 网页横幅只用于 GitHub；PDF 使用统一的讲义封面。
    text = re.sub(r'<p\b[^>]*>\s*<img\b[^>]*mlp-cover\.svg[^>]*>\s*</p>', "", text)
    # HTML 折叠框在纸面展开，摘要成为真正的章节标题。
    text = re.sub(r"<summary>(.*?)</summary>", r"\n\n## \1\n\n", text, flags=re.S)
    text = re.sub(r"</?details\b[^>]*>", "", text)
    # 手记中独占一行的单个 $ 按跨行公式处理，原始 MD 不变。
    text = re.sub(r"(?m)^([ \t]*)\$[ \t]*\n([\s\S]*?)\n\1\$[ \t]*$",
                  lambda match: match[1] + "$$\n" + match[2] + "\n" + match[1] + "$$", text)
    # 原稿有 "$公式$1." 等紧接编号的写法；在定界符外补空格，
    # 避免 Pandoc 将公式末尾的美元符号误读为正文，原始 MD 不变。
    text = re.sub(r"(?<!\$)\$(?!\$)([^\n$]+?)\$(?!\$)",
                  lambda match: " $" + match[1] + "$ ", text)
    return text


def verify_images(document, source):
    count = 0

    def visit(value):
        nonlocal count
        if isinstance(value, dict):
            if value.get("t") == "Image":
                target = value["c"][2][0]
                if re.match(r"https?://", target):
                    raise RuntimeError(f"请先将远程图片保存到仓库：{target}")
                if not (source.parent / target).is_file():
                    raise RuntimeError(f"图片不存在：{source.parent / target}")
                count += 1
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(document)
    return count


def build_markdown(source, engine):
    # 临时目录由上下文管理器清理，成功或中断都不在仓库留下编译产物。
    with TemporaryDirectory(prefix="markdown-pdf-") as temporary:
        compile_markdown(source, engine, Path(temporary))


def compile_markdown(source, engine, work):
    target = source.with_suffix(".pdf")
    print(f"输出目标：{target}", flush=True)
    pandoc = executable("pandoc")
    result = subprocess.run(
        [pandoc, "--from=markdown+tex_math_dollars-implicit_figures", "--to=json"],
        input=prepare_markdown(source), text=True, encoding="utf-8", capture_output=True, check=True,
    )
    document = json.loads(result.stdout)
    # 从解析后的标题取封面文字，避免将代码块中的 # 误认为标题。
    heading = next((block for block in document["blocks"] if block["t"] == "Header"), None)
    if heading:
        plain = subprocess.run(
            [pandoc, "--from=json", "--to=plain", "--wrap=none"],
            input=json.dumps({**document, "blocks": [heading]}),
            text=True, encoding="utf-8", capture_output=True, check=True,
        )
        title = plain.stdout.strip()
    else:
        title = source.stem
    image_count = verify_images(document, source)
    tex = work / "lecture.tex"
    subprocess.run(
        [pandoc, "--from=json", "--to=latex", "--standalone", "--syntax-highlighting=none",
         f"--template={SCRIPTS / 'lecture-template.tex'}",
         f"--lua-filter={SCRIPTS / 'lecture-layout.lua'}",
         "--metadata", f"title={title}", "--metadata", "subtitle=学习笔记",
         "--metadata", f"shorttitle={title}", "--metadata", f"source={source.as_posix()}",
         "--toc", "--toc-depth=3", "--output", str(tex)],
        input=result.stdout, text=True, encoding="utf-8", check=True,
    )
    # 在原 Markdown 的目录运行，以保留全部相对图片路径。
    command = [engine, "--disable-installer", "-interaction=nonstopmode", "-halt-on-error",
               "-file-line-error", f"-output-directory={work}", str(tex)]
    previous = None
    for run in range(1, 4):
        print(f"{source.name}: XeLaTeX {run}/3，{image_count} 幅图", flush=True)
        result = subprocess.run(command, cwd=source.parent, capture_output=True)
        if result.returncode:
            tail = result.stdout.decode("utf-8", errors="replace")[-4000:]
            raise RuntimeError(f"编译中断，未更新最终 PDF：{target}\n{tail}")
        state = tuple((work / f"lecture{ext}").read_bytes()
                      if (work / f"lecture{ext}").exists() else b""
                      for ext in (".aux", ".toc", ".out"))
        if run >= 2 and state == previous:
            break
        previous = state
    log = (work / "lecture.log").read_text(encoding="utf-8", errors="replace")
    issues = [line for line in log.splitlines()
              if "Missing character:" in line or "Overfull \\hbox" in line
              or "Overfull \\vbox" in line or "undefined references" in line]
    shutil.copy2(work / "lecture.pdf", target)
    print(f"已生成 {target}", flush=True)
    if issues:
        print("排版警告（PDF 已输出）：\n" + "\n".join(issues),
              file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, metavar="Markdown路径",
                        help="Markdown 文件的相对或绝对路径；生成或覆盖同名 PDF")
    parser.add_argument("--engine", help="XeLaTeX 的可执行文件路径")
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    if source.suffix.lower() not in {".md", ".markdown"}:
        parser.error("请输入 .md 或 .markdown 文件路径。")
    if not source.is_file():
        parser.error(f"Markdown 文件不存在：{source}")
    engine = args.engine or executable("xelatex", "D:/miktex/miktex/bin/x64/xelatex.exe")
    build_markdown(source, engine)
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(error, file=sys.stderr)
        raise SystemExit(1)
