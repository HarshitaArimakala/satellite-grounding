import os
from datasets import load_dataset
from PIL import Image

IMAGES_DIR = "data/images/Images_val"  # adjust if unzip created a subfolder — check with `ls` above

dataset = load_dataset("xiang709/VRSBench", split="validation", streaming=True)
samples = list(dataset.take(15))

os.makedirs("data/sample", exist_ok=True)
for i, ex in enumerate(samples):
    img_path = os.path.join(IMAGES_DIR, ex["image"])
    img = Image.open(img_path)
    img.save(f"data/sample/img_{i}.png")

print("Saved", len(samples), "samples")
