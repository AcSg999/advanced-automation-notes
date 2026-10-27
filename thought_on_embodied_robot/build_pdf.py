"""从同目录 Markdown 重建中文小书。依赖 reportlab、matplotlib、Pillow、pypdf。"""
from pathlib import Path
import os
import re
import sys
import html

# 本机可直接复用 Codex 已有的纯 Python 排版库；其他电脑正常安装依赖即可。
bundled = Path.home() / '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/Lib/site-packages'
if bundled.exists():
    sys.path.append(str(bundled))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
from matplotlib.textpath import TextPath
from matplotlib.path import Path as MplPath
from matplotlib.font_manager import FontProperties
from PIL import Image as PILImage
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.lib.units import mm
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph,
    Spacer, PageBreak, CondPageBreak, Image, Table, TableStyle, KeepTogether, Flowable)
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'assets'
ASSETS.mkdir(exist_ok=True)
MD = ROOT / '具身智能技术路线小书.md'
PDF = ROOT / '具身智能技术路线小书.pdf'
FONT = os.environ.get('BOOK_FONT', 'C:/Windows/Fonts/simsun.ttc')
BOLD = os.environ.get('BOOK_FONT_BOLD', 'C:/Windows/Fonts/simhei.ttf')
if not Path(FONT).is_file():
    raise SystemExit('请用 BOOK_FONT 指定中文 TrueType 字体文件，并用 BOOK_FONT_BOLD 指定粗体文件。')
if not Path(BOLD).is_file():
    BOLD = FONT
pdfmetrics.registerFont(TTFont('Book', FONT, subfontIndex=0))
pdfmetrics.registerFont(TTFont('BookBold', BOLD, subfontIndex=0))
pdfmetrics.registerFontFamily('Book', normal='Book', bold='BookBold', italic='Book', boldItalic='BookBold')
for name,filename in [('Latin','times.ttf'),('LatinBold','timesbd.ttf'),('LatinItalic','timesi.ttf'),('LatinBoldItalic','timesbi.ttf')]:
    pdfmetrics.registerFont(TTFont(name, str(Path('C:/Windows/Fonts')/filename)))
pdfmetrics.registerFontFamily('Latin',normal='Latin',bold='LatinBold',italic='LatinItalic',boldItalic='LatinBoldItalic')
# 六幅原有教学图仍保留无衬线图中文字；正文独立使用宋体与衬线英文。
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family': font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),
                     'axes.unicode_minus': False, 'font.size': 12, 'savefig.facecolor': 'white',
                     'mathtext.fontset':'cm'})
INK = '#17324D'
TEAL = '#087F8C'
GOLD = '#C68A39'
LIGHT = '#EDF4F6'
GRAY = '#536878'


def canvas(w=10, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, h)
    ax.axis('off')
    return fig, ax


def box(ax, x, y, w, h, title, sub='', color=TEAL):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.04,rounding_size=0.1',
        linewidth=1.2, edgecolor=color, facecolor=LIGHT))
    ax.text(x+w/2, y+h*.65 if sub else y+h/2, title, ha='center', va='center',
        color=INK, fontsize=12, fontweight='bold')
    if sub:
        ax.text(x+w/2, y+h*.27, sub, ha='center', va='center', color=GRAY, fontsize=10)


def arrow(ax, start, end, label=None, color=TEAL):
    ax.annotate('', xy=end, xytext=start, arrowprops={'arrowstyle':'->', 'color':color, 'lw':1.6})
    if label:
        ax.text((start[0]+end[0])/2+.08, (start[1]+end[1])/2, label, color=GRAY, fontsize=9)


def save(fig, name):
    fig.savefig(ASSETS/name, dpi=220, bbox_inches='tight', pad_inches=.16)
    plt.close(fig)


def diagrams():
    fig, ax = canvas(h=6.2)
    layers = [('目标与任务结构', 'LLM / VLM、行为树、任务规划'),
              ('状态与记忆', '视觉、三维、地图、任务进度'),
              ('预测与动作生成', 'VLA、ACT、扩散、WAM、MPC'),
              ('快速运动与接触反馈', '轨迹跟踪、阻抗、力触觉、全身控制'),
              ('身体与环境', '执行器、物体、接触、空间')]
    for i,(title,sub) in enumerate(layers):
        y=5.05-i*1.06
        box(ax,1.45,y,7.1,.83,title,sub)
        if i<4:
            arrow(ax,(5,y),(5,y-.22))
    ax.plot([8.8,9.25,9.25,8.8],[.42,.42,5.48,5.48], color=GOLD, lw=1.8)
    arrow(ax,(9.25,5.48),(8.75,5.48),color=GOLD)
    ax.text(9.45,2.95,'反\n馈\n与\n验\n证',ha='center',va='center',color=GOLD,fontsize=12)
    ax.text(.7,3,'慢\n↓\n快',ha='center',va='center',color=GRAY)
    save(fig,'01_stack.png')

    fig, ax=canvas(h=4.4)
    ax.text(.2,4,'H = 8 个预测动作；m = 3 个执行动作（示例）',fontsize=13,color=INK)
    for row,start in enumerate([0,3,6]):
        y=2.9-row*.95
        ax.text(.1,y+.23,f'观测 {start}',va='center',color=INK,fontsize=11)
        for j in range(8):
            x=1.6+(start+j)*.54
            ax.add_patch(Rectangle((x,y),.48,.48,facecolor=TEAL if j<3 else LIGHT,edgecolor=TEAL,lw=.8))
            ax.text(x+.24,y+.24,str(start+j),ha='center',va='center',fontsize=9,
                    color='white' if j<3 else GRAY)
        if row<2:
            arrow(ax,(1.6+(start+3)*.54,y-.05),(1.6+(start+3)*.54,y-.42))
    ax.text(1.6,.13,'深色：本轮执行　　浅色：预测后尚未执行；新观测到来后重算',fontsize=10,color=GRAY)
    save(fig,'02_chunks.png')

    fig,ax=canvas(h=5.5)
    box(ax,.25,3.8,2.65,1,'视觉与任务','目标位姿、接近方向')
    box(ax,3.65,3.8,2.65,1,'策略与阶段判断','接近 / 接触 / 退出')
    box(ax,7.05,3.8,2.65,1,'柔顺执行','位置、力、阻抗目标')
    arrow(ax,(2.95,4.3),(3.58,4.3));arrow(ax,(6.35,4.3),(6.98,4.3))
    box(ax,7.05,1.65,2.65,1,'插头与插孔','接触、摩擦、卡滞')
    box(ax,3.65,1.65,2.65,1,'力与触觉历史','受力、力矩、滑移')
    arrow(ax,(8.37,3.75),(8.37,2.72));arrow(ax,(7,2.15),(6.38,2.15))
    arrow(ax,(4.97,2.72),(4.97,3.75))
    ax.plot([6.35,6.7,6.7],[2.15,2.15,4.32],color=GOLD,lw=1.5)
    arrow(ax,(6.7,4.32),(6.98,4.32),color=GOLD)
    ax.text(.35,2.15,'视觉给出远处目标\n接触揭示局部误差\n低层及时响应',color=GRAY,fontsize=12,va='center',linespacing=1.9)
    ax.text(3.75,.7,'金色支路：快速反馈可直接进入控制层',color=GOLD,fontsize=11)
    save(fig,'03_contact.png')

    fig,ax=canvas(h=5.1)
    rows=[('在线规划','候选动作 → 预测后果 → 评分 → 执行开头'),
          ('想象训练','真实数据 → 潜模型 → 想象轨迹 → 训练策略'),
          ('联合生成','当前观测与目标 → 未来状态 + 动作块')]
    for i,(title,sub) in enumerate(rows):
        y=3.65-i*1.35
        box(ax,.3,y,2.05,.9,title)
        box(ax,2.95,y,6.65,.9,sub,color=GOLD if i==2 else TEAL)
        arrow(ax,(2.4,y+.45),(2.88,y+.45))
    ax.text(.4,.1,'从在线规划、想象训练与联合生成三个位置理解世界模型。',fontsize=11,color=GRAY)
    save(fig,'04_world.png')

    fig,ax=plt.subplots(figsize=(9.4,4.9))
    ns=list(range(1,41))
    for p,color in [(.95,INK),(.99,TEAL)]:
        ax.plot(ns,[p**n for n in ns],label=f'单步成功 p = {p}',lw=2,color=color)
    p=.95
    ax.plot(ns,[(1-(1-p)**2)**n for n in ns],label='p = 0.95，每步允许一次独立重试',lw=2,color=GOLD)
    ax.set_xlabel('连续关键步骤数 n');ax.set_ylabel('全部完成的概率')
    ax.set_ylim(0,1.03);ax.set_xlim(1,40)
    ax.grid(alpha=.18);ax.legend(loc='lower left',fontsize=10,frameon=False)
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();save(fig,'05_reliability.png')

    fig,ax=canvas(h=6.25)
    pairs=[('语义与目标不清楚','VLM / VLA + 对象落地'),
           ('空间关系或可达性错误','三维感知 + 规划 / 约束'),
           ('接触卡滞、滑移、受力未知','力触觉 + 柔顺反馈'),
           ('长流程失去进度或条件冲突','记忆 + 任务结构 + 验证'),
           ('动作后果不可预期','世界模型 / WAM + 真反馈'),
           ('快速运动或身体失稳','RL / MPC + 全身控制')]
    for i,(left,right) in enumerate(pairs):
        y=5.25-i*.88
        box(ax,.25,y,4.05,.66,left)
        box(ax,5.05,y,4.65,.66,right,color=GOLD if i==2 else TEAL)
        arrow(ax,(4.35,y+.33),(4.98,y+.33))
    ax.text(.3,.05,'先诊断失败机制，再选组件；多个瓶颈可以同时存在。',fontsize=11,color=GRAY)
    save(fig,'06_selection.png')


def math_path(expr, size=12):
    expr=expr.replace(r'\text{execute }',r'\mathrm{execute}\;')
    expr=re.sub(r'\\(mathcal|mathbb)\s+([A-Za-z])',r'\\\1{\2}',expr)
    expr=re.sub(r'\\le(?![A-Za-z])',r'\\leq ',expr)
    expr=re.sub(r'\\ge(?![A-Za-z])',r'\\geq ',expr)
    return TextPath((0,0),'$'+expr+'$',size=size,prop=FontProperties(family='serif'))


def paint_path(c, path, x, y):
    q=c.beginPath(); last=(0,0)
    for verts,code in path.iter_segments(curves=True,simplify=False):
        if code==MplPath.MOVETO:
            last=(verts[0],verts[1]);q.moveTo(x+last[0],y+last[1])
        elif code==MplPath.LINETO:
            last=(verts[0],verts[1]);q.lineTo(x+last[0],y+last[1])
        elif code==MplPath.CURVE3:
            cx,cy,ex,ey=verts
            q.curveTo(x+last[0]+2*(cx-last[0])/3,y+last[1]+2*(cy-last[1])/3,
                      x+ex+2*(cx-ex)/3,y+ey+2*(cy-ey)/3,x+ex,y+ey)
            last=(ex,ey)
        elif code==MplPath.CURVE4:
            q.curveTo(*[v+(x if j%2==0 else y) for j,v in enumerate(verts)])
            last=(verts[4],verts[5])
        elif code==MplPath.CLOSEPOLY:q.close()
    c.drawPath(q,fill=1,stroke=0,fillMode=0)


class MathFormula(Flowable):
    """以矢量字形绘制 Computer Modern 公式，避免将公式放大成位图。"""
    def __init__(self, expr):
        super().__init__()
        self.matrix=r'\begin{bmatrix}' in expr
        if self.matrix:
            self.parts=[(math_path(r'T=\mathrm{FK}(q)=',12),0,0)]
            for x,y,t in [(109,8,'R'),(136,8,'p'),(109,-10,'0'),(136,-10,'1')]:
                self.parts.append((math_path(t,12),x,y))
            self.natural_width=157;self.natural_height=40;self.bottom=-17
        else:
            self.path=math_path(expr,12)
            bb=self.path.get_extents()
            self.natural_width=bb.width+6;self.natural_height=bb.height+10
            self.bottom=bb.ymin
            self.left=bb.xmin
        self.hAlign='CENTER'
    def wrap(self,availWidth,availHeight):
        self.scale=min(1,availWidth/self.natural_width)
        self.width=self.natural_width*self.scale
        self.height=self.natural_height*self.scale
        return self.width,self.height
    def draw(self):
        c=self.canv;c.saveState();c.scale(self.scale,self.scale);c.setFillColor(colors.black)
        if self.matrix:
            for path,x,y in self.parts:paint_path(c,path,x,y+21)
            c.setStrokeColor(colors.black);c.setLineWidth(.55)
            for x,sign in [(101,-1),(151,1)]:
                q=c.beginPath();q.moveTo(x,37);q.lineTo(x+sign*3,37);q.lineTo(x+sign*3,3);q.lineTo(x,3)
                c.drawPath(q)
        else:paint_path(c,self.path,3-self.left,5-self.bottom)
        c.restoreState()


W,H=210*mm,297*mm
M=25*mm
CW=W-2*M
BLACK=colors.HexColor('#111111')
styles={
    'body':ParagraphStyle('body',fontName='Book',fontSize=10.8,leading=17,
        textColor=BLACK,alignment=TA_JUSTIFY,wordWrap='CJK',firstLineIndent=21.6,
        spaceAfter=3.5,allowWidows=0,allowOrphans=0),
    'h1':ParagraphStyle('h1',fontName='Book',fontSize=16,leading=23,textColor=BLACK,
        wordWrap='CJK',spaceBefore=18,spaceAfter=12,keepWithNext=True),
    'h2':ParagraphStyle('h2',fontName='Book',fontSize=12.2,leading=19,textColor=BLACK,
        wordWrap='CJK',spaceBefore=11,spaceAfter=5,keepWithNext=True),
    'number':ParagraphStyle('number',fontName='Book',fontSize=10.8,leading=17,textColor=BLACK,
        wordWrap='CJK',leftIndent=16,firstLineIndent=-16,spaceAfter=3),
    'caption':ParagraphStyle('caption',fontName='Book',fontSize=9.5,leading=15,textColor=BLACK,
        alignment=TA_CENTER,wordWrap='CJK',spaceBefore=6,spaceAfter=11),
    'quote':ParagraphStyle('quote',fontName='Book',fontSize=10.2,leading=16,textColor=BLACK,
        wordWrap='CJK',leftIndent=12,rightIndent=12,spaceAfter=12),
    'cell':ParagraphStyle('cell',fontName='Book',fontSize=9.2,leading=14,textColor=BLACK,wordWrap='CJK'),
    'th':ParagraphStyle('th',fontName='BookBold',fontSize=9.2,leading=14,textColor=BLACK,wordWrap='CJK'),
    'ref':ParagraphStyle('ref',fontName='Book',fontSize=9.4,leading=14.8,textColor=BLACK,
        wordWrap='CJK',leftIndent=25,firstLineIndent=-25,spaceAfter=7,splitLongWords=True),
}


def inline(s, refs=True):
    s=html.escape(s)
    s=re.sub(r'`([^`]+)`',r'<font name="Latin">\1</font>',s)
    s=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',s)
    s=re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)',r'<link href="\2">\1</link>',s)
    if refs:
        s=re.sub(r'\[(R\d{2})\]',r'<super><link href="#\1">[\1]</link></super>',s)
    out=[];bold=False;infont=False
    for part in re.split(r'(<[^>]+>|&[A-Za-z0-9#]+;)',s):
        if part.startswith('&'):
            out.append(part);continue
        if part.startswith('<'):
            if part=='<b>':bold=True
            elif part=='</b>':bold=False
            elif part.startswith('<font '):infont=True
            elif part=='</font>':infont=False
            out.append(part);continue
        if infont:out.append(part);continue
        def latin(m):
            base,sub=m.group(1),m.group(2)
            name='LatinBold' if bold else 'Latin'
            mathvar=len(base)==1 and base.isalpha()
            if mathvar:name='LatinBoldItalic' if bold else 'LatinItalic'
            rendered=f'<font name="{name}">{base}</font>'
            if sub:
                rendered+=f'<sub><font name="LatinItalic">{sub}</font></sub>'
            return rendered
        part=re.sub(r'([A-Za-z0-9α-ωΑ-Ω]+)(?:_([A-Za-z0-9α-ωΑ-Ω]+|[≤≥][A-Za-z0-9]+))?',latin,part)
        part=part.replace('ᵀ','<super><font name="LatinItalic">T</font></super>')
        part=part.replace('²','<super>2</super>')
        out.append(part)
    return ''.join(out)


class SongHeading(Paragraph):
    def draw(self):
        c=self.canv;c.saveState();c.setLineWidth(.16)
        t=c.beginText();t.setTextRenderMode(2);c.drawText(t)
        super().draw();c.restoreState()


class BookDoc(BaseDocTemplate):
    def afterFlowable(self, flowable):
        if isinstance(flowable,Paragraph) and getattr(flowable,'chapter_key',None):
            key=flowable.chapter_key
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(flowable.getPlainText(),key,0,False)
            self.notify('TOCEntry',(0,flowable.getPlainText(),self.page,key))


def page(c,doc):
    c.setFillColor(colors.black)
    if doc.page==1:
        c.setFont('Book',25)
        c.drawCentredString(W/2,H-61*mm,'具身智能技术路线')
        c.setFont('Book',13)
        c.drawCentredString(W/2,H-77*mm,'从感知、接触与决策到基础模型')
        c.setFont('Latin',11)
        c.drawCentredString(W/2,H-91*mm,'VLA  ·  VFA  ·  LLM  ·  WAM')
        c.setFont('Book',10.5)
        c.drawCentredString(W/2,H-107*mm,'2026 年 10 月')
        c.setFont('Book',9.5)
        c.drawCentredString(W/2,43*mm,'技术综述与原理讲解 · 讲义修订版')
        c.setFont('Latin',10)
        c.drawCentredString(W/2,18*mm,'1')
        return
    c.setFont('Book',9)
    c.drawString(M,H-15*mm,'具身智能技术路线')
    c.setFont('Latin',10)
    c.drawRightString(W-M,H-15*mm,str(doc.page))


def image_flow(path, maxh=105*mm, maxw=CW, maxscale=float('inf')):
    with PILImage.open(path) as im:
        iw,ih=im.size
    scale=min(maxw/iw,maxh/ih,maxscale)
    return Image(str(path),width=iw*scale,height=ih*scale,hAlign='CENTER')


def make_pdf():
    if not all((ASSETS/f'{name}.png').is_file() for name in ['01_stack','02_chunks','03_contact','04_world','05_reliability','06_selection']):
        diagrams()
    text=MD.read_text(encoding='utf-8')
    # 封面信息由页面模板生成；正文从阅读定位开始。
    lines=text[text.index('> 阅读定位'):].splitlines()
    toc=TableOfContents()
    toc.levelStyles=[ParagraphStyle('toc0',fontName='Book',fontSize=10.5,leading=18,
        leftIndent=0,firstLineIndent=0,rightIndent=20,spaceBefore=2,textColor=BLACK,wordWrap='CJK')]
    toc.tableStyle=TableStyle([('LEFTPADDING',(0,0),(-1,-1),0),
        ('RIGHTPADDING',(0,0),(-1,-1),0),('TOPPADDING',(0,0),(-1,-1),0),
        ('BOTTOMPADDING',(0,0),(-1,-1),0),('VALIGN',(0,0),(-1,-1),'TOP')])
    toc.dotsMinLevel=0
    story=[Spacer(1,10),PageBreak(),SongHeading('目录',styles['h1']),toc,PageBreak()]
    i=0; eq=0; chapter=0; inrefs=False; started=False
    while i<len(lines):
        line=lines[i].strip()
        if not line:
            i+=1;continue
        if line=='$$':
            j=i+1;expr=[]
            while j<len(lines) and lines[j].strip()!='$$':
                expr.append(lines[j].strip());j+=1
            eq+=1
            story.extend([Spacer(1,4),MathFormula(' '.join(expr)),Spacer(1,5)])
            i=j+1;continue
        if line.startswith('# '):
            if started:
                story.extend([CondPageBreak(76*mm),Spacer(1,10)])
            started=True;chapter+=1
            title=re.sub(r'^第 (\d+) 章[　 ]*',r'\1　',line[2:])
            inrefs=title=='参考资料'
            p=SongHeading(inline(title),styles['h1']);p.chapter_key=f'chapter{chapter}'
            story.append(p);i+=1;continue
        if line.startswith('## '):
            title=line[3:]
            if title.startswith('导读'):
                started=True;chapter+=1
                p=SongHeading(inline(title),styles['h1']);p.chapter_key=f'chapter{chapter}'
            else:p=SongHeading(inline(title),styles['h2'])
            story.append(p);i+=1;continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                row=[x.strip() for x in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?',x.replace(' ','')) for x in row):rows.append(row)
                i+=1
            n=len(rows[0]);weights={2:[.34,.66],3:[.23,.37,.40],4:[.19,.27,.27,.27],5:[.16,.21,.21,.21,.21]}.get(n,[1/n]*n)
            vpad=5
            pdata=[[Paragraph(inline(s),styles['th' if r==0 else 'cell']) for s in row] for r,row in enumerate(rows)]
            tab=Table(pdata,colWidths=[CW*w for w in weights],repeatRows=1,hAlign='LEFT')
            tab.setStyle(TableStyle([('LINEABOVE',(0,0),(-1,0),.8,colors.black),
                ('LINEBELOW',(0,0),(-1,0),.45,colors.black),
                ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),
                ('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),vpad),
                ('BOTTOMPADDING',(0,0),(-1,-1),vpad),('LINEBELOW',(0,-1),(-1,-1),.8,colors.black)]))
            # 表格与紧邻的小节标题一起分页，避免标题留在上一页底部。
            if story and isinstance(story[-1],Paragraph) and story[-1].style.name=='h2':
                heading=story.pop()
                story.append(KeepTogether([heading,Spacer(1,5),tab,Spacer(1,12)]))
            else:
                story.extend([Spacer(1,5),KeepTogether([tab]),Spacer(1,12)])
            continue
        match=re.match(r'!\[([^\]]*)\]\(([^)]+)\)',line)
        if match:
            story.append(KeepTogether([image_flow(ROOT/match[2]),Paragraph(inline(match[1]),styles['caption'])]))
            i+=1;continue
        if line.startswith('> '):
            story.append(Paragraph(inline(line[2:]),styles['quote']));i+=1;continue
        if inrefs and line.startswith('- [R'):
            rid=re.match(r'- \[(R\d+)\]',line)[1]
            content=line[2:]
            story.append(Paragraph(f'<a name="{rid}"/>'+inline(content,refs=False),styles['ref']))
            i+=1;continue
        if re.match(r'\d+\. ',line):
            items=[]
            while i<len(lines) and re.match(r'^\d+\. ',lines[i].strip()):
                items.append(Paragraph(inline(lines[i].strip()),styles['number']))
                i+=1
            if story and isinstance(story[-1],Paragraph) and story[-1].style.name=='h2':
                items.insert(0,story.pop())
            story.append(KeepTogether(items));continue
        parts=[line];i+=1
        while i<len(lines) and lines[i].strip() and not re.match(r'^(#|\||!\[|\$\$|>|- |\d+\. )',lines[i].strip()):
            parts.append(lines[i].strip());i+=1
        story.append(Paragraph(inline(' '.join(parts)),styles['body']))
    doc=BookDoc(str(PDF),pagesize=(W,H),leftMargin=M,rightMargin=M,topMargin=26*mm,
        bottomMargin=24*mm,title='具身智能技术路线小书',author='技术综述与原理讲解',
        subject='VLA、VFA、LLM、WAM 与具身任务解决路线',pageCompression=1)
    frame=Frame(M,24*mm,CW,H-50*mm,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
    doc.addPageTemplates(PageTemplate(id='book',frames=[frame],onPage=page))
    doc.multiBuild(story)
    from pypdf import PdfReader
    r=PdfReader(PDF)
    print(f'完成：{PDF.name} | {len(r.pages)} 页 | {eq} 个公式 | 6 幅图 | {chapter} 个目录条目')


if __name__=='__main__':
    make_pdf()
