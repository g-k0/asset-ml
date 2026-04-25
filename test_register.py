import psycopg2
import io
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import psycopg2

# Setup same as api.py
clip_model_id = "openai/clip-vit-base-patch32"
clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
clip_model = CLIPModel.from_pretrained(clip_model_id)
SUPABASE_URI = "postgresql://postgres.gkksyrejaiqpxchcyvrb:UYLh%2Aq%24b7ayZMR%2B@aws-1-ap-south-1.pooler.supabase.com:5432/postgres"

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

# Simulate what register_art endpoint does
async def register_art(artist_name, file_filename, image_bytes):
    print(f"Registering new art for {artist_name}: {file_filename}...")
    
    try:
        # 1. Open the image
        img = Image.open(io.BytesIO(image_bytes))
        
        # Get the exact middle
        width, height = img.size
        mid_x, mid_y = width // 2, height // 2
        
        # 2. Chop into 5 pieces
        regions = {
            "full_image": img,
            "top_left": img.crop((0, 0, mid_x, mid_y)),
            "top_right": img.crop((mid_x, 0, width, mid_y)),
            "bottom_left": img.crop((0, mid_y, mid_x, height)),
            "bottom_right": img.crop((mid_x, mid_y, width, height))
        }
        
        # 3. Connect to Supabase
        conn = psycopg2.connect(SUPABASE_URI)
        cursor = conn.cursor()
        
        # 4. Loop through pieces and save vectors
        for region_name, crop_img in regions.items():
            print(f"Processing {region_name}...")
            
            # Save cropped image to bytes
            img_byte_arr = io.BytesIO()
            crop_img.save(img_byte_arr, format=img.format or 'JPEG')
            crop_bytes = img_byte_arr.getvalue()
            
            # Get CLIP vector
            vector = get_clip_vector(crop_bytes)
            print(f"  -> Vector length: {len(vector)}")
            
            vector_string = "[" + ",".join(map(str, vector)) + "]"
            
            # Insert into Supabase
            cursor.execute("""
                INSERT INTO protected_assets (artist_name, image_name, region, visual_dna)
                VALUES (%s, %s, %s, %s);
            """, (artist_name, file_filename, region_name, vector_string))
            
            print(f"  -> Saved vector for: {region_name}")

        # 5. Commit
        conn.commit()
        cursor.close()
        conn.close()
        
        print("Successfully locked 5 vectors into the Supabase Vault!")
        return {"status": "success", "message": f"Secured {file_filename} across 5 regions."}
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

# Run the test
if __name__ == "__main__":
    with open("diddy.jpeg", "rb") as f:
        image_bytes = f.read()
    
    import asyncio
    result = asyncio.run(register_art("Test Artist", "diddy.jpeg", image_bytes))
    print(f"Result: {result}")