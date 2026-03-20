"""
Run this script ONCE on your local machine inside the project folder.
It will load all 4 HuggingFace datasets + builtin, save to server/cache/,
then repackage everything as a zip ready for HF Spaces submission.

Usage:
    cd hallucination_guard_env_final
    python generate_cache.py
"""

import sys, os, json, zipfile, shutil
sys.path.insert(0, '.')
sys.path.insert(0, './server')

print("=" * 55)
print("  HallucinationGuard-Env  —  Cache Generator")
print("=" * 55)
print()

from server.dataset_loader import DatasetLoader

loader = DatasetLoader()

# Step 1: builtin
print("[1/3] Loading builtin synthetic dataset...")
n1 = loader.load_builtin_datasets()
print(f"      ✅ {n1} examples loaded")
print()

# Step 2: HuggingFace datasets
print("[2/3] Loading HuggingFace datasets (may take 1-2 min first time)...")
n2 = loader.load_real_datasets(max_per_dataset=500, cache=True)
print(f"      ✅ {n2} examples loaded from HF")
print()

# Step 3: stats
stats = loader.statistics
print("[3/3] Final statistics:")
print(f"      Total examples : {stats.total_examples}")
print(f"      By difficulty  : {stats.examples_by_difficulty}")
print(f"      By source      : {stats.examples_by_source}")
print(f"      Avg context    : {stats.average_context_length:.0f} chars")
print()

# Check cache file was written
cache_dir = "server/cache"
cache_files = os.listdir(cache_dir)
if cache_files:
    for f in cache_files:
        size = os.path.getsize(os.path.join(cache_dir, f))
        print(f"      Cache file: {f}  ({size/1024:.0f} KB)")
    print()
    print("✅ Cache saved! Now packaging zip...")
    print()
else:
    print("❌ Cache file not found — something went wrong.")
    sys.exit(1)

# Package the zip
project_dir = os.path.abspath(".")
parent_dir  = os.path.dirname(project_dir)
zip_path    = os.path.join(parent_dir, "hallucination_guard_env_SUBMISSION.zip")

EXCLUDE = {"__pycache__", ".git", "*.pyc", "metrics_logs"}

with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(project_dir):
        # skip unwanted dirs
        dirs[:] = [d for d in dirs if d not in EXCLUDE]
        for file in files:
            if file.endswith(".pyc"):
                continue
            abs_path = os.path.join(root, file)
            arc_path = os.path.relpath(abs_path, parent_dir)
            zf.write(abs_path, arc_path)

size_mb = os.path.getsize(zip_path) / (1024 * 1024)
print(f"✅ Submission zip created!")
print(f"   Path : {zip_path}")
print(f"   Size : {size_mb:.1f} MB")
print()
print("Next steps:")
print("  1. Upload to HuggingFace Spaces:  openenv push --repo-id SamSankar/hallucination-guard-env")
print("  2. Submit the HF Spaces URL on the hackathon dashboard before April 7, 11:59 PM IST")
print()
print("Good luck Sam! 🔥")
