import os, glob, numpy as np
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(PROJECT_ROOT, 'dataset_root')

def main():
    print('=' * 80)
    print('AUDITING RE-EXTRACTED SENTINEL-1 SAR CONTINUOUS ARRAYS')
    print('=' * 80)
    for split in ['train', 'val', 'test']:
        sar_dir = os.path.join(DATASET_ROOT, split, 'sar')
        if not os.path.isdir(sar_dir):
            continue
        files = sorted(glob.glob(os.path.join(sar_dir, '*.npy')))
        if not files:
            continue
        std_list, unique_list = [], []
        sampled = files[:20] if len(files) >= 20 else files
        for f in sampled:
            arr = np.load(f)
            vv = arr[0]
            std_list.append(float(vv.std()))
            unique_list.append(len(np.unique(vv)))
        mean_std = float(np.mean(std_list))
        mean_unique = float(np.mean(unique_list))
        min_unique = int(np.min(unique_list))
        print(f'[{split.upper()}] Patches on disk: {len(files)} | Mean Std: {mean_std:.4f} | Mean Unique: {mean_unique:.0f} (Min: {min_unique})')
        status = 'PASS - Genuine Radar Speckle' if min_unique > 20000 else 'Quantized'
        print(f'  Status: {status}')

if __name__ == '__main__':
    main()
