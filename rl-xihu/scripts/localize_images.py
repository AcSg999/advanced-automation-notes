from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen
from PIL import Image
import hashlib
import json
import re
import shutil
import time

root = Path(__file__).resolve().parent.parent
note = next(root.glob('*.md'))
backup = root / 'backup'
backup.mkdir(exist_ok=True)
original = backup / note.name
if not original.exists():
    shutil.copy2(note, original)
source = original.read_text(encoding='utf-8')
urls = list(dict.fromkeys(re.findall(r'!\[[^\]]*\]\((https?://[^)]+)\)', source)))
images = root / 'images'
images.mkdir(exist_ok=True)

def download(item):
    index, url = item
    extension = Path(url.split('?')[0]).suffix.lower()
    path = images / f'{index:03d}{extension}'
    for attempt in range(3):
        try:
            if not path.exists():
                request = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urlopen(request, timeout=45) as response:
                    data = response.read()
                temporary = path.with_suffix(path.suffix + '.part')
                temporary.write_bytes(data)
                with Image.open(temporary) as image:
                    image.verify()
                temporary.replace(path)
            with Image.open(path) as image:
                image.verify()
            return {'index': index, 'source_url': url,
                    'local_path': path.relative_to(root).as_posix(),
                    'bytes': path.stat().st_size,
                    'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        except Exception:
            if attempt == 2:
                raise
            time.sleep(attempt + 1)

results = []
with ThreadPoolExecutor(max_workers=8) as executor:
    futures = {executor.submit(download, item): item for item in enumerate(urls, 1)}
    for future in as_completed(futures):
        results.append(future.result())
        if len(results) % 20 == 0:
            print(f'Downloaded and verified {len(results)}/{len(urls)}', flush=True)
results.sort(key=lambda item: item['index'])
(images / 'manifest.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
current = note.read_text(encoding='utf-8')
for item in results:
    current = current.replace('](' + item['source_url'] + ')', '](' + item['local_path'] + ')')
note.write_text(current, encoding='utf-8')
print(f'Localized {len(results)} images; total bytes: {sum(x["bytes"] for x in results)}', flush=True)
