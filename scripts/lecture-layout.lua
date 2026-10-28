-- 图片随正文原位排版，避免浮动导致课件截图与相邻解释分离。
local function split_images(block)
  local output = pandoc.List()
  local text = pandoc.List()
  local function flush()
    if #text > 0 then
      output:insert(pandoc.Para(text))
      text = pandoc.List()
    end
  end
  local found = false
  for _, inline in ipairs(block.content) do
    if inline.t == 'Image' then
      found = true
      -- 将图片与同段中紧邻的解释一起换页，避免页底留下孤立说明。
      local has_text = pandoc.utils.stringify(text):match('%S') ~= nil
      if has_text then
        output:insert(pandoc.RawBlock('latex', '\\par\\noindent\\begin{minipage}{\\linewidth}'))
      end
      flush()
      local path = inline.src:gsub('\\', '/')
      output:insert(pandoc.RawBlock('latex',
        '\\begin{center}\\includegraphics[width=0.96\\linewidth,' ..
        'height=0.39\\textheight,keepaspectratio]{\\detokenize{' .. path ..
        '}}\\end{center}'))
      if has_text then
        output:insert(pandoc.RawBlock('latex', '\\end{minipage}\\par\\smallskip'))
      end
    else
      text:insert(inline)
    end
  end
  if not found then return nil end
  flush()
  return output
end

function CodeBlock(block)
  return pandoc.RawBlock('latex', '\\par\\noindent\\begin{minipage}{\\linewidth}\n' ..
    '\\begin{lstlisting}\n' .. block.text .. '\n\\end{lstlisting}\n\\end{minipage}\\par')
end

function Header(block)
  -- 三份原稿的主要章节均从二级标题开始。
  block.level = math.max(1, block.level - 1)
  return block
end

return {{Para = split_images, Plain = split_images, CodeBlock = CodeBlock, Header = Header}}
