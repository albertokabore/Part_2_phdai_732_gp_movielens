"""Render Albert's notebook analysis into the draft without changing results/.

Run from the repository root. Uses committed tuning surfaces, refits evaluation
models, and exports the notebook's plots and tables into a detailed appendix.
"""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
if (ROOT / '.draft_deps').exists():
    sys.path.insert(0, str(ROOT / '.draft_deps'))
os.environ.setdefault('SURPRISE_DATA_FOLDER', str(ROOT / 'data'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'drafts' / '.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

OUT = ROOT / 'drafts' / 'notebook_figures'
OUT.mkdir(exist_ok=True)
notebook = json.loads((ROOT / 'notebooks/part2_tuning_evaluation.ipynb').read_text(encoding='utf-8'))
draft = ROOT / 'drafts/albert_tuning_and_evaluation.md'
marker = '\n## Detailed Notebook Analysis and Visualizations\n'
base = draft.read_text(encoding='utf-8').split(marker)[0].rstrip()
before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / 'results').glob('*') if p.is_file()}
parts = [marker, '\nThis expanded appendix includes every numbered visualization group (V1–V17) and the supporting analysis for Albert’s Hyperparameter Tuning and Final Model Evaluation sections. It is intentionally longer than the assignment’s short report allocation. Figure labels match the notebook; the six original report figures remain in the main sections above.\n',
         '\nTuning plots use the committed `tuning_surface_*.csv` files. Evaluation plots are rendered from the notebook’s model fits on the frozen splits and checked against the committed results. For V16, tuned catalog lists come directly from `final_top10_all_users.csv`, keeping exposure counts consistent with the report; the untuned lists are recomputed. Near-tied catalog scores can produce slightly different lists across numerical-library builds. Supplemental tables are saved beside the images. These exports do not replace the group’s authoritative `results/` files.\n']
log = io.StringIO()
current = 0
plot_number = 0
table_number = 0
manifest = []

def table_markdown(frame):
    frame = frame.copy()
    if not isinstance(frame.index, pd.RangeIndex):
        frame = frame.reset_index()
    def fmt(v):
        if isinstance(v, float):
            return '' if pd.isna(v) else f'{v:.6g}'
        return str(v).replace('|', '\\|').replace('\n', ' ')
    rows = ['| ' + ' | '.join(map(fmt, frame.columns)) + ' |', '| ' + ' | '.join(['---'] * len(frame.columns)) + ' |']
    rows.extend('| ' + ' | '.join(fmt(v) for v in row) + ' |' for row in frame.itertuples(index=False, name=None))
    return '\n'.join(rows)

def display(value):
    global table_number
    if isinstance(value, pd.Series):
        value = value.to_frame()
    if isinstance(value, pd.DataFrame):
        table_number += 1
        name = f'cell_{current:02d}_table_{table_number:02d}.csv'
        value.to_csv(OUT / name)
        if current != 66:
            parts.append('\n' + table_markdown(value) + '\n')
        else:
            parts.append(f'\n[Complete numerical claim audit](notebook_figures/{name})\n')

def show(*args, **kwargs):
    global plot_number
    for num in plt.get_fignums():
        fig = plt.figure(num)
        plot_number += 1
        titles = [ax.get_title() for ax in fig.axes if ax.get_title()]
        title = fig._suptitle.get_text() if fig._suptitle else '; '.join(titles)
        name = f'cell_{current:02d}_figure_{plot_number:02d}.png'
        fig.savefig(OUT / name, dpi=180, bbox_inches='tight', facecolor='white')
        parts.append(f'\n**{title}**\n\n![{title}](notebook_figures/{name})\n')
        manifest.append({'cell': current, 'title': title, 'file': name})
        plt.close(fig)

plt.show = show
ns = {'display': display, '__name__': '__draft_export__'}
# Markdown already represented by original report figures, setup, or summary.
skip_markdown = {0, 1, 12, 16, 22, 37, 44, 61, 64}
code_titles = {8: 'Assignment candidates', 14: 'Expanded search results', 20: 'Refined search results',
               25: 'Selection across all three searches', 28: 'Detailed sensitivity and candidate checks',
               31: 'Test and training error comparison', 35: 'Prediction summary by actual rating',
               42: 'Error by training support', 47: 'Largest individual errors',
               50: 'Ranking metrics and list-length comparisons', 55: 'Catalog exposure summaries',
               59: 'Chronological split comparison', 63: 'Report tables and saved-result checks', 66: 'Numerical claim audit'}
for current, cell in enumerate(notebook['cells']):
    source = ''.join(cell['source'])
    if cell['cell_type'] == 'markdown':
        if current not in skip_markdown:
            source = re.sub(r'^(#{2,3}) ', lambda m: '#' + m.group(1) + ' ', source, flags=re.M)
            parts.append('\n' + source + '\n')
        continue
    if current == 2:
        continue
    source = '\n'.join(line for line in source.splitlines() if not line.startswith('%'))
    source = source.replace('RUN_FULL_SEARCH = True', 'RUN_FULL_SEARCH = False')
    source = source.replace('surf_assign = run_search(param_grid)', 'surf_assign = load_surface("assignment")')
    if current == 55:
        source = source.replace('top = top_n_from_scores(algo.trainset, scores, n=C.TOP_K)',
                                'top = pd.read_csv(RES / "final_top10_all_users.csv") if name == "Tuned SVD" else top_n_from_scores(algo.trainset, scores, n=C.TOP_K)')
    if current in code_titles:
        parts.append('\n#### ' + code_titles[current] + '\n')
    print(f'Rendering notebook cell {current}', flush=True)
    with contextlib.redirect_stdout(log):
        exec(compile(source, f'notebook cell {current}', 'exec'), ns)

assert len(manifest) == 18, f'Expected 18 images for V1–V17 (V5 has two), got {len(manifest)}'
assert not (ns['audit'].status != 'OK').any(), 'Numerical claim audit failed'
assert 'DIFF' not in log.getvalue(), 'Saved-result comparison failed; inspect rendering log'
assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in before.items()), 'A results file changed'
parts.append(f'\nAll {len(ns["audit"])} numerical claim checks passed. All saved-result comparisons passed the notebook’s stated tolerances. The tuning searches were loaded from saved surfaces; evaluation models were refit. CV error bars describe fold variability and are not significance tests.\n')
(OUT / 'render_log.txt').write_text(log.getvalue(), encoding='utf-8')
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
draft.write_text(base + '\n' + '\n'.join(parts), encoding='utf-8')

# Keep the Word and HTML handoffs synchronized.
from render_word_draft import render as render_word
from render_draft_html import render as render_html
render_word()
render_html()
print(f'Exported {len(manifest)} notebook images and {table_number} tables.')
