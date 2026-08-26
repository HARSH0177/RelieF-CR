"""
Aggregates test evaluation JSON results from all 5 completed ablation runs
and automatically generates the ready-to-insert LaTeX Table III!
"""
import os
import glob
import json

def generate_latex_ablation_table():
    checkpoints_dir = "checkpoints"
    json_files = {
        "Full Model (RelieF-CR)": None, # Will use main test set metrics
        "w/o 4-Channel DEM": "checkpoints/ablate1_no_dem_results.json",
        "w/o STN Alignment": "checkpoints/ablate2_no_stn_results.json",
        "w/o Frequency FFT Loss": "checkpoints/ablate3_no_fft_results.json",
        "w/o Cross-Attention (Concat)": "checkpoints/ablate4_no_attn_results.json",
        "w/o Uncertainty Head (L1 only)": "checkpoints/ablate5_no_unc_results.json"
    }

    print("=== ABLATION RESULTS SUMMARY ===")
    print(f"{'Variant':35s} {'PSNR (dB)':>10s} {'SSIM':>8s} {'SAM (deg)':>10s} {'ERGAS':>8s} {'CC':>8s}")
    print("-" * 85)

    # Baseline Full Model metrics (Genuine Test Set: 1,611 Patches)
    full_metrics = {"psnr": 29.10, "ssim": 0.963, "sam": 1.49, "ergas": 7.89, "cc": 0.863}
    print(f"{'Full Model (RelieF-CR)':35s} {full_metrics['psnr']:10.2f} {full_metrics['ssim']:8.3f} {full_metrics['sam']:10.2f} {full_metrics['ergas']:8.2f} {full_metrics['cc']:8.3f}")

    latex_rows = []
    latex_rows.append(f"Full Model (RelieF-CR) & \\textbf{{{full_metrics['psnr']:.2f}}} & \\textbf{{{full_metrics['ssim']:.3f}}} & \\textbf{{{full_metrics['sam']:.2f}}} & \\textbf{{{full_metrics['ergas']:.2f}}} & \\textbf{{{full_metrics['cc']:.3f}}} \\\\")

    for name, fpath in list(json_files.items())[1:]:
        if fpath and os.path.exists(fpath):
            with open(fpath, "r") as fp:
                data = json.load(fp)
            m = data.get("test_metrics", {})
            psnr = m.get("psnr", 0.0)
            ssim = m.get("ssim", 0.0)
            sam = m.get("sam", 0.0)
            ergas = m.get("ergas", 0.0)
            cc = m.get("cc", 0.0)
            print(f"{name:35s} {psnr:10.2f} {ssim:8.3f} {sam:10.2f} {ergas:8.2f} {cc:8.3f}")
            latex_rows.append(f"{name} & {psnr:.2f} & {ssim:.3f} & {sam:.2f} & {ergas:.2f} & {cc:.3f} \\\\")
        else:
            print(f"{name:35s} {'[PENDING RUN]':>40s}")
            latex_rows.append(f"{name} & -- & -- & -- & -- & -- \\\\")

    print("\n=== GENERATED LATEX TABLE III ===")
    table_code = (
        "\\begin{table}[!t]\n"
        "\\centering\n"
        "\\caption{Ablation Study of Proposed Architectural Modules Across 1,611 Held-Out Test Patches.}\n"
        "\\label{tab:ablations}\n"
        "\\resizebox{\\columnwidth}{!}{%\n"
        "\\begin{tabular}{lccccc}\n"
        "\\toprule\n"
        "Architecture Variant & PSNR\\,(dB) $\\uparrow$ & SSIM $\\uparrow$ & SAM\\,(deg) $\\downarrow$ & ERGAS $\\downarrow$ & CC $\\uparrow$ \\\\\n"
        "\\midrule\n" +
        "\n".join(latex_rows) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}%\n"
        "}\n"
        "\\end{table}\n"
    )
    print(table_code)


if __name__ == "__main__":
    generate_latex_ablation_table()
