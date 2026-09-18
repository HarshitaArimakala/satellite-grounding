import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

device = "mps" if torch.backends.mps.is_available() else "cpu"

model_id = "IDEA-Research/grounding-dino-tiny"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

print("Loaded on:", device)
