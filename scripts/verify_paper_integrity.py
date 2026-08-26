import re
import glob
import os

def audit_paper():
    bib_keys = set()
    with open('paper/references.bib', 'r', encoding='utf-8') as f:
        for line in f:
            m = re.match(r'@\w+\s*\{\s*([^,]+),', line.strip())
            if m:
                bib_keys.add(m.group(1).strip())

    print(f'Found {len(bib_keys)} BibTeX keys in references.bib')

    labels = set()
    cites = set()
    refs = set()

    for fpath in sorted(glob.glob('paper/**/*.tex', recursive=True)):
        with open(fpath, 'r', encoding='utf-8') as f:
            txt = f.read()
        
        # Check citations
        for c in re.findall(r'\\cite\{([^}]+)\}', txt):
            for k in c.split(','):
                k = k.strip()
                cites.add(k)
                if k not in bib_keys:
                    print(f'ERROR: Missing citation key "{k}" in {fpath}')
        
        # Check labels
        for l in re.findall(r'\\label\{([^}]+)\}', txt):
            labels.add(l.strip())
        
        # Check refs
        for r in re.findall(r'\\ref\{([^}]+)\}', txt):
            refs.add(r.strip())

    print(f'Total unique citations: {len(cites)}')
    missing_cites = cites - bib_keys
    if not missing_cites:
        print('ALL CITATIONS RESOLVE: PASS')
    else:
        print('MISSING CITATIONS:', missing_cites)

    missing_refs = refs - labels
    if not missing_refs:
        print('ALL LABELS AND REFS RESOLVE: PASS')
    else:
        print('MISSING REFS:', missing_refs)

if __name__ == '__main__':
    audit_paper()
