"""从 Markdown 重建讲义 PDF；具身智能讲义沿用原生 LaTeX 编译。"""

from pathlib import Path
import argparse
import json
import re
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
LECTURES = {
    "knn": ("deep_learning/KNN.md", "KNN：K 近邻方法", "从样本间的距离到分类与回归"),
    "mlp": ("deep_learning/mlp.md", "MLP：多层感知机", "前向传播、反向传播与梯度更新"),
    "rl": ("rl-xihu/西湖大学强化学习.md", "西湖大学强化学习", "课程学习笔记：从 Bellman 方程到 Actor-Critic"),
}
EMBODIED = ROOT / "thought_on_embodied_robot/roadmap_for_embodied_tech.pdf"


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
    if source.stem == "西湖大学强化学习":
        text = re.sub(r"\A\s*# 西湖大学强化学习\s*\n", "", text)
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


def build_markdown(key, engine):
    relative, title, subtitle = LECTURES[key]
    source = ROOT / relative
    target = source.with_suffix(".pdf")
    work = ROOT / ".build" / "pdf" / key
    work.mkdir(parents=True, exist_ok=True)
    pandoc = executable("pandoc")
    result = subprocess.run(
        [pandoc, "--from=markdown+tex_math_dollars-implicit_figures", "--to=json"],
        input=prepare_markdown(source), text=True, encoding="utf-8", capture_output=True, check=True,
    )
    document = json.loads(result.stdout)
    image_count = verify_images(document, source)
    tex = work / "lecture.tex"
    subprocess.run(
        [pandoc, "--from=json", "--to=latex", "--standalone", "--syntax-highlighting=none",
         f"--template={SCRIPTS / 'lecture-template.tex'}",
         f"--lua-filter={SCRIPTS / 'lecture-layout.lua'}",
         "--metadata", f"title={title}", "--metadata", f"subtitle={subtitle}",
         "--metadata", f"shorttitle={title}", "--metadata", f"source={relative}",
         "--toc", "--toc-depth=1", "--output", str(tex)],
        input=result.stdout, text=True, encoding="utf-8", check=True,
    )
    # 在原 Markdown 的目录运行，以保留全部相对图片路径。
    command = [engine, "--disable-installer", "-interaction=nonstopmode", "-halt-on-error",
               "-file-line-error", f"-output-directory={work}", str(tex)]
    previous = None
    for run in range(1, 4):
        print(f"{key}: XeLaTeX {run}/3，{image_count} 幅图", flush=True)
        result = subprocess.run(command, cwd=source.parent, capture_output=True)
        if result.returncode:
            tail = result.stdout.decode("utf-8", errors="replace")[-4000:]
            raise RuntimeError(f"编译失败：{work / 'lecture.log'}\n{tail}")
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
    if issues:
        raise RuntimeError(f"需要检查版面：{work / 'lecture.log'}\n" + "\n".join(issues))
    shutil.copy2(work / "lecture.pdf", target)
    # 只清理本脚本在仓库 .build/pdf 下生成的目录。
    if work.resolve().is_relative_to((ROOT / ".build" / "pdf").resolve()):
        shutil.rmtree(work)
    print(f"已生成 {target.relative_to(ROOT)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lectures", nargs="*", metavar="主题",
                        help="knn / mlp / rl / embodied；指定主题时重建该 PDF")
    parser.add_argument("--all", action="store_true", help="重建全部 4 份 PDF")
    parser.add_argument("--engine", help="XeLaTeX 的可执行文件路径")
    args = parser.parse_args()
    unknown = set(args.lectures) - {*LECTURES, "embodied"}
    if unknown:
        parser.error(f"未知主题：{', '.join(sorted(unknown))}")
    targets = list(dict.fromkeys(args.lectures))
    if args.all:
        targets = [*LECTURES, "embodied"]
    elif not targets:
        targets = [key for key, (relative, _, _) in LECTURES.items()
                   if not (ROOT / relative).with_suffix(".pdf").is_file()]
        if not EMBODIED.is_file():
            targets.append("embodied")
    if not targets:
        print("4 份 PDF 均已存在。修改正文后可指定主题重建，或使用 --all。")
        return 0
    engine = args.engine or executable("xelatex", "D:/miktex/miktex/bin/x64/xelatex.exe")
    for key in targets:
        if key == "embodied":
            subprocess.run([sys.executable, str(EMBODIED.parent / "build_pdf.py"),
                            "--engine", engine], check=True)
        else:
            build_markdown(key, engine)
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
