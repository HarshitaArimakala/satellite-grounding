import json
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor

# Only the processor is needed for Step 1 (no model, so this is fast)
processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")

# Load the first sample from the file build_samples.py made
s = json.load(open("data/sample/samples.json"))[0]
image = Image.open(Path("data/sample") / s["image"]).convert("RGB")

query = s["query"].lower().strip()
if not query.endswith("."):
    query += "."

print("Query:", query)
print("Original image size (W, H):", image.size)

# ---- the actual Step 1 ----
inputs = processor(images=image, text=query, return_tensors="pt")

print("\n=== What the processor produced ===")
print("keys:", list(inputs.keys()))

# ---- Part A: the image ----
pv = inputs.pixel_values
print("\n=== Image side ===")
print("pixel_values shape [batch, channels, H, W]:", tuple(pv.shape))
print(f"value range after normalizing: min {pv.min():.2f}, max {pv.max():.2f}, mean {pv.mean():.2f}")
print("raw top-left pixel (RGB, 0-255):", np.array(image)[0, 0])
print("normalized top-left pixel (R, G, B):", pv[0, :, 0, 0])
# these two won't match exactly, because the image was resized (pixels get blended)
# but the normalized one should be small numbers around -2 to +2
if "pixel_mask" in inputs:
    print("pixel_mask all ones (no padding)?", bool(inputs.pixel_mask.all()))

# ---- Part B: the sentence ----
ids = inputs.input_ids[0]
print("\n=== Text side ===")
print("token IDs:", ids.tolist())
print("tokens   :", processor.tokenizer.convert_ids_to_tokens(ids))
print("attention_mask:", inputs.attention_mask[0].tolist())