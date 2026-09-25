"""Create a standalone, illustrated HTML copy of Albert's Markdown draft."""
import base64
import html
from pathlib import Path
import re

import markdown


def render():
    folder = Path(__file__).resolve().parent
    source = folder / 'albert_tuning_and_evaluation.md'
    text = source.read_text(encoding='utf-8')
    images = []

    def embed(match):
        title, target = match.groups()
        image = (folder / target).resolve()
        if not image.is_file():
            raise FileNotFoundError(image)
        data = base64.b64encode(image.read_bytes()).decode('ascii')
        images.append(target)
        return f'![{title}](data:image/png;base64,{data})'

    text = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', embed, text)
    converter = markdown.Markdown(extensions=['tables', 'fenced_code', 'toc'])
    body = converter.convert(text)
    title = "Albert's Draft — Tuning and Evaluation"
    page = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>''' + html.escape(title) + '''</title>
<style>
body{margin:0;background:#f4f6f8;color:#192330;font:17px/1.65 Georgia,serif}
main{max-width:1200px;margin:auto;padding:36px;background:white}
h1,h2,h3,h4,nav{font-family:system-ui,sans-serif;line-height:1.3}
h2{margin-top:3rem;border-bottom:2px solid #dae3ed;padding-bottom:.5rem}
img{display:block;width:auto;max-width:100%;height:auto;margin:24px auto}
table{display:block;overflow-x:auto;border-collapse:collapse;font-size:14px;margin:20px 0}
th,td{border:1px solid #d6dde5;padding:8px 12px;text-align:left}
th{background:#eaf1f9}a{color:#165aa0}pre{overflow:auto;background:#f1f4f7;padding:16px}
nav{background:#edf4fc;padding:18px 24px;border-radius:8px;font-size:15px}
nav ul{padding-left:22px}section.notice{background:#edf4fc;padding:16px 24px;margin-bottom:24px}
@media print{body{background:white}main{padding:0}nav,.notice{display:none}img{break-inside:avoid}h2,h3,h4{break-after:avoid}}
</style></head><body><main>
<section class="notice"><strong>All charts are embedded in this file.</strong>
The draft includes six report figures and 18 notebook images covering V1–V17
(V5 uses two images). Use the contents below to jump to the detailed notebook analysis.
You can also use your browser's Print command to save a PDF.</section>
<nav aria-label="Contents"><strong>Contents</strong>''' + converter.toc + '</nav>' + body + '</main></body></html>'
    output = folder / 'albert_tuning_and_evaluation.html'
    output.write_text(page, encoding='utf-8')
    assert page.count('<img ') == len(images)
    print(f'Created {output.name}: {len(images)} embedded images.')
    return output


if __name__ == '__main__':
    render()
