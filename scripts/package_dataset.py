import os
import zipfile
import time
from concurrent.futures import ThreadPoolExecutor

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(PROJECT_ROOT, 'dataset_root')
ARCHIVE_DIR = os.path.join(PROJECT_ROOT, 'dataset_archives')

os.makedirs(ARCHIVE_DIR, exist_ok=True)

def zip_split(split_name):
    src_dir = os.path.join(DATASET_ROOT, split_name)
    if not os.path.isdir(src_dir):
        print(f'Directory not found: {src_dir}')
        return
    
    zip_path = os.path.join(ARCHIVE_DIR, f'{split_name}.zip')
    print(f'Starting archiving for [{split_name}] -> {zip_path}...')
    t0 = time.time()
    
    all_files = []
    for root, _, filenames in os.walk(src_dir):
        for f in filenames:
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, DATASET_ROOT)
            all_files.append((full_path, rel_path))
            
    print(f'[{split_name}] Found {len(all_files)} files to archive...')
    
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for idx, (full_path, rel_path) in enumerate(all_files):
            zf.write(full_path, arcname=rel_path)
            if (idx + 1) % 5000 == 0 or (idx + 1) == len(all_files):
                pct = (idx + 1) / len(all_files) * 100
                elapsed = time.time() - t0
                print(f'[{split_name}] {idx + 1}/{len(all_files)} files ({pct:.1f}%) archived in {elapsed:.1f}s...', flush=True)
                
    elapsed = time.time() - t0
    size_gb = os.path.getsize(zip_path) / (1024**3)
    print(f'[{split_name}] COMPLETE! Size: {size_gb:.2f} GB in {elapsed:.1f}s')

def main():
    print('=' * 80)
    print('PACKAGING DATASET INTO COMPRESSED ZIP ARCHIVES')
    print('=' * 80)
    
    # Process val and test first (faster), then train
    for split in ['val', 'test', 'train']:
        zip_split(split)
        
    print('=' * 80)
    print(f'ALL ARCHIVES READY IN: {ARCHIVE_DIR}')
    print('=' * 80)

if __name__ == '__main__':
    main()
