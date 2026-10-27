"""编译同目录的原生 LaTeX 讲义：python build_pdf.py。"""
from pathlib import Path
import argparse
import shutil
import subprocess
import sys

# VS Code 的输出面板按 UTF-8 解码编译日志。
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tex', default='roadmap_for_embodied_tech.tex', help='要编译的 tex 文件')
    parser.add_argument('--engine', help='XeLaTeX 可执行文件路径')
    args = parser.parse_args()
    tex = (ROOT / args.tex).resolve()
    if not tex.is_file():
        parser.error(f'找不到正文：{tex}')
    engine = args.engine or shutil.which('xelatex')
    if not engine and Path('D:/miktex/miktex/bin/x64/xelatex.exe').is_file():
        engine = 'D:/miktex/miktex/bin/x64/xelatex.exe'
    if not engine:
        parser.error('找不到 xelatex，请将现有 MiKTeX 或 TeX Live 的 bin 目录加入 PATH。')
    out = ROOT / '.build'
    out.mkdir(exist_ok=True)
    command = [engine, '--disable-installer', '-synctex=1', '-interaction=nonstopmode',
               '-halt-on-error', '-file-line-error', '-output-directory=' + str(out), str(tex)]
    # 目录、交叉引用和页码需经过多轮编译。正常编译始终读取现有 tex。
    previous = None
    for run in range(1, 4):
        print(f'XeLaTeX {run}/3: {tex.name}', flush=True)
        result = subprocess.run(command, cwd=ROOT)
        if result.returncode:
            print(f'编译失败。请查看 {out / (tex.stem + ".log")}', file=sys.stderr)
            return result.returncode
        state = tuple((out / (tex.stem + ext)).read_bytes() if (out / (tex.stem + ext)).exists() else b''
                      for ext in ('.aux', '.toc', '.out'))
        if run >= 2 and state == previous:
            break
        previous = state
    target = ROOT / (tex.stem + '.pdf')
    try:
        shutil.copy2(out / target.name, target)
    except PermissionError:
        print(f'根目录 PDF 正被其他程序占用；新 PDF 已生成在 {out / target.name}', file=sys.stderr)
        return 1
    print(f'PDF: {target}', flush=True)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
