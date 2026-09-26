import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

#AutoProcessor (conv img/text to numbers) (Pre-processing)
#AutoModelForZeroShotObjectDetection (Grounding-DINO) (Encoders and Cross-Attention)

device = "mps" if torch.backends.mps.is_available() else "cpu"

model_id = "IDEA-Research/grounding-dino-tiny"
processor = AutoProcessor.from_pretrained(model_id) #pre-processing config
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device) #loads the pre-trained weights for grounding-DINO

print("Loaded on:", device)
