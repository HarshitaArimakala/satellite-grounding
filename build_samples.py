import json, re, random, shutil
from pathlib import Path
from PIL import Image

IMG_DIR = Path("data/images/Images_val")
OUT_DIR = Path("data/sample"); OUT_DIR.mkdir(exist_ok=True)

records = json.load(open("data/VRSBench_EVAL_referring.json"))
print(len(records), "records. Example:", records[0])

# Sanity check: what range do the box numbers live in?
nums = [int(n) for r in records for n in re.findall(r"<(\d+)>", r["ground_truth"])]
print("box number range:", min(nums), "to", max(nums))

random.seed(0)       
random.shuffle(records)

samples, seen = [], set()
for r in records:
    if r["image_id"] in seen or not (IMG_DIR / r["image_id"]).exists():
        continue           
    coords = [int(n) for n in re.findall(r"<(\d+)>", r["ground_truth"])]
    if len(coords) != 4:
        continue
    seen.add(r["image_id"])
    W, H = Image.open(IMG_DIR / r["image_id"]).size
    x1, y1, x2, y2 = coords
    box = [x1/100*W, y1/100*H, x2/100*W, y2/100*H]   # ASSUMES 0-100 scale, see check above
    shutil.copy(IMG_DIR / r["image_id"], OUT_DIR / r["image_id"])
    samples.append({"image": r["image_id"], "query": r["question"], "box": box})
    if len(samples) == 15:
        break

json.dump(samples, open(OUT_DIR / "samples.json", "w"), indent=2)
print("saved", len(samples), "samples")
