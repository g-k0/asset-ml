import sys
import json
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# 1. Load OpenAI's CLIP Model (Incredibly accurate for image understanding)
# It downloads the model automatically the first time you run it.
model_id = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id)

def extract_ai_fingerprint(image_path):
    try:
        # 2. Open the image
        image = Image.open(image_path).convert("RGB")
        
        # 3. Process the image for the neural network
        inputs = processor(images=image, return_tensors="pt")
        
        # 4. Run the image through the network (No gradient calculation needed)
        with torch.no_grad():
            outputs = model(**inputs)
            image_features = outputs.image_embeds[0].flatten()
            
        # 5. Convert the tensor into a standard list of numbers
        vector = image_features.tolist()
        
        # Return as a JSON string so Node.js can read it easily
        print(json.dumps({"status": "success", "vector": vector}))
        
    except Exception as e:
        print(json.dumps({"status": "error", "message": str(e)}))

if __name__ == "__main__":
    # This allows us to pass the image path from the terminal command
    if len(sys.argv) > 1:
        extract_ai_fingerprint(sys.argv[1])
    else:
        print(json.dumps({"status": "error", "message": "No image path provided."}))