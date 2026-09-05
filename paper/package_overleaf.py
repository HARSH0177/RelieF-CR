import os
import zipfile
import shutil
from PIL import Image

paper_dir = r"C:\Users\HARSH AMBULE\Downloads\cloudfree_vision_v2_extracted\cloudfree_vision_v2\paper"
out_zip_path = r"C:\Users\HARSH AMBULE\Downloads\paper_overleaf.zip"

# If an old paper zip exists in Downloads, remove it
for old_f in ["paper.zip", "paper_overleaf.zip"]:
    op = os.path.join(r"C:\Users\HARSH AMBULE\Downloads", old_f)
    if os.path.exists(op):
        try:
            os.remove(op)
            print(f"Removed old zip: {op}")
        except Exception as e:
            print(f"Could not remove {op}: {e}")

staging_dir = os.path.join(paper_dir, "_overleaf_staging")
if os.path.exists(staging_dir):
    shutil.rmtree(staging_dir)
os.makedirs(staging_dir, exist_ok=True)

# 1. Copy main.tex and references.bib
shutil.copy2(os.path.join(paper_dir, "main.tex"), os.path.join(staging_dir, "main.tex"))
shutil.copy2(os.path.join(paper_dir, "references.bib"), os.path.join(staging_dir, "references.bib"))

# 2. Copy sections
staging_sec = os.path.join(staging_dir, "sections")
os.makedirs(staging_sec, exist_ok=True)
for f in os.listdir(os.path.join(paper_dir, "sections")):
    if f.endswith(".tex"):
        shutil.copy2(os.path.join(paper_dir, "sections", f), os.path.join(staging_sec, f))

# 3. Copy diagrams
staging_diag = os.path.join(staging_dir, "diagrams")
os.makedirs(staging_diag, exist_ok=True)
for f in os.listdir(os.path.join(paper_dir, "diagrams")):
    if f.endswith(".tex"):
        shutil.copy2(os.path.join(paper_dir, "diagrams", f), os.path.join(staging_diag, f))

# 4. Process and copy referenced figures
staging_fig = os.path.join(staging_dir, "figures")
os.makedirs(staging_fig, exist_ok=True)

referenced_figures = [
    "cloudfree_vision_v2_architecture.png",
    "comparison_grid_trained.png",
    "study_area_map.png",
    "training_convergence_sensitivity.png",
    "sar_gradient_correlation_distribution.png",
    "uncertainty_calibration.png"
]

for rfig in referenced_figures:
    src_fp = os.path.join(paper_dir, "figures", rfig)
    dst_fp = os.path.join(staging_fig, rfig)
    if os.path.exists(src_fp):
        # Optimize PNG using PIL without quality loss
        try:
            img = Image.open(src_fp)
            # Save optimized PNG
            img.save(dst_fp, "PNG", optimize=True)
            sz_kb = os.path.getsize(dst_fp) / 1024
            print(f"  Optimized {rfig}: {sz_kb:.1f} KB")
        except Exception as e:
            shutil.copy2(src_fp, dst_fp)
            print(f"  Copied {rfig} directly: {os.path.getsize(dst_fp)/1024:.1f} KB")

# 5. Create zip archive
print(f"\nCreating {out_zip_path}...")
with zipfile.ZipFile(out_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(staging_dir):
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, staging_dir)
            zf.write(full_path, rel_path)
            print(f"  + Added: {rel_path} ({os.path.getsize(full_path)/1024:.1f} KB)")

# Clean up staging
shutil.rmtree(staging_dir)

total_zip_mb = os.path.getsize(out_zip_path) / (1024 * 1024)
print(f"\nSuccessfully created Overleaf Zip: {out_zip_path}")
print(f"Total Package Size: {total_zip_mb:.2f} MB (Ultra lightweight, fits Overleaf!)")
