"""Apply rewrite_map.json to the docx — clean and simple."""
import sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from docx import Document

with open('report/rewrite_map.json', 'r', encoding='utf-8') as f:
    mapping = json.load(f)

doc = Document('report/实验报告.docx')

changed = 0
mismatches = 0
for i, p in enumerate(doc.paragraphs):
    key = str(i)
    if key in mapping:
        old, new = mapping[key]
        current = p.text.strip()
        if current == old:
            # Replace text in first run, clear rest
            runs = p.runs
            if runs:
                runs[0].text = new
                for r in runs[1:]:
                    r.text = ""
            changed += 1
            print(f'[OK] p{i}')
        else:
            mismatches += 1
            print(f'[SKIP] p{i}: text mismatch (len: {len(current)} vs {len(old)})')
            # Show first diff
            for j, (a, b) in enumerate(zip(current, old)):
                if a != b:
                    print(f'  First diff at pos {j}: "{current[max(0,j-5):j+10]}" vs "{old[max(0,j-5):j+10]}"')
                    break

doc.save('report/实验报告.docx')
print(f'\nDone: {changed} changed, {mismatches} skipped, {len(mapping)} total mappings')
