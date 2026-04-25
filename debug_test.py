import psycopg2
import io
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

clip_model_id = "openai/clip-vit-base-patch32"
clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
clip_model = CLIPModel.from_pretrained(clip_model_id)

def get_clip_vector(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    dummy_text = [" "]
    text_inputs = clip_processor(text=dummy_text, return_tensors="pt", padding=True)
    inputs["input_ids"] = text_inputs["input_ids"]
    inputs["attention_mask"] = text_inputs["attention_mask"]
    
    with torch.no_grad():
        outputs = clip_model(**inputs)
        clip_vector = outputs.image_embeds[0].flatten().tolist()
    return clip_vector

# Test CLIP
print("Testing CLIP vector extraction...")
img = Image.new('RGB', (100, 100), color='red')
img_bytes = io.BytesIO()
img.save(img_bytes, format='JPEG')
img_bytes.seek(0)

try:
    vector = get_clip_vector(img_bytes.getvalue())
    print(f"CLIP vector length: {len(vector)}")
except Exception as e:
    print(f"CLIP Error: {e}")
    import traceback
    traceback.print_exc()

# Supabase connection
SUPABASE_URI = "postgresql://postgres.gkksyrejaiqpxchcyvrb:UYLh%2Aq%24b7ayZMR%2B@aws-1-ap-south-1.pooler.supabase.com:5432/postgres"

print("\nTesting Supabase INSERT...")
try:
    conn = psycopg2.connect(SUPABASE_URI)
    cursor = conn.cursor()
    
    vector = get_clip_vector(img_bytes.getvalue())
    vector_string = "[" + ",".join(map(str, vector)) + "]"
    
    cursor.execute("""
        INSERT INTO protected_assets (artist_name, image_name, region, visual_dna)
        VALUES (%s, %s, %s, %s);
    """, ('test_artist', 'test.jpg', 'full_image', vector_string))
    
    conn.commit()
    print("INSERT successful!")
    cursor.close()
    conn.close()
except Exception as e:
    print(f"INSERT Error: {e}")
    import traceback
    traceback.print_exc()