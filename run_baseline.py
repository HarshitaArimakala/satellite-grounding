import os, json
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import sys
import torch
from pathlib import Path
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

device = "mps" if torch.backends.mps.is_available() else "cpu"
model_id = "IDEA-Research/grounding-dino-tiny"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()

def iou(a, b):                                   # boxes are [x1, y1, x2, y2]
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0

def predict(image, query):
    """Return (best_box, score) or (None, 0) if nothing passes the threshold."""
    query = query.lower().strip()
    if not query.endswith("."):
        query += "."
    inputs = processor(images=image, text=query, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    res = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=0.05, text_threshold=0.1,
        target_sizes=[image.size[::-1]])[0]
    if len(res["boxes"]) == 0:
        return None, 0.0
    best = res["scores"].argmax()
    return res["boxes"][best].tolist(), res["scores"][best].item()

def load_samples():
    return json.load(open("data/sample/samples.json"))   # [{"image":..., "query":..., "box":[x1,y1,x2,y2]}, ...]

Path("outputs").mkdir(exist_ok=True)
ious = []
for i, s in enumerate(load_samples()):
    image = Image.open(Path("data/sample") / s["image"]).convert("RGB")
    pred, score = predict(image, s["query"])
    val = iou(pred, s["box"]) if pred else 0.0
    ious.append(val)
    print(f"{i:2d}  IoU={val:.2f}  score={score:.2f}  {s['query'][:60]}")

    d = ImageDraw.Draw(image)
    d.rectangle(s["box"], outline="lime", width=3)          # ground truth
    if pred: d.rectangle(pred, outline="red", width=3)      # prediction
    image.save(f"outputs/{i:02d}_iou{val:.2f}.png")

print(f"\nMean IoU: {sum(ious)/len(ious):.3f}")
print(f"Acc@0.5:  {sum(v >= 0.5 for v in ious)/len(ious):.2f}")