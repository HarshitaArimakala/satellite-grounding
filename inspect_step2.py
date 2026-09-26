import json, torch
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

model_id = "IDEA-Research/grounding-dino-tiny"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).eval()   # CPU is fine here

# See what the model is made of (use these names if the attributes below fail)
print("top-level parts:", [n for n, _ in model.named_children()])
print("inside model.model:", [n for n, _ in model.model.named_children()])

s = json.load(open("data/sample/samples.json"))[0]
image = Image.open(Path("data/sample") / s["image"]).convert("RGB")
query = s["query"].lower().strip()
query = query if query.endswith(".") else query + "."
inputs = processor(images=image, text=query, return_tensors="pt")

with torch.no_grad():
    # ---- image encoder ----
    features, _ = model.model.backbone(inputs.pixel_values, inputs.pixel_mask)
    print("\n=== IMAGE ENCODER ===")
    print("input pixel_values:", tuple(inputs.pixel_values.shape))
    for i, (fmap, _mask) in enumerate(features):
        print(f"level {i}: feature map shape {tuple(fmap.shape)}  "
            f"-> {fmap.shape[2]}x{fmap.shape[3]} grid, {fmap.shape[1]} numbers per cell")

    # ---- text encoder ----
    text_out = model.model.text_backbone(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        token_type_ids=inputs.get("token_type_ids"),
    )
    tf = text_out.last_hidden_state
    print("\n=== TEXT ENCODER ===")
    print("tokens:", processor.tokenizer.convert_ids_to_tokens(inputs.input_ids[0]))
    print("text features shape [batch, tokens, numbers]:", tuple(tf.shape))
    print("first token vector, first 5 numbers:", tf[0, 0, :5])