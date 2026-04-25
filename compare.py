import requests
import math

url = 'http://localhost:8000/analyze-asset'

def get_ai_vector(image_path):
    print(f"📡 Scanning {image_path}...")
    try:
        with open(image_path, 'rb') as file:
            response = requests.post(url, files={'file': file})
            data = response.json()
            vector = data.get('similarity_vector')
            
            # THE BULLETPROOF FIX: Keep peeling off layers until we hit raw numbers!
            while isinstance(vector, list) and len(vector) > 0 and isinstance(vector[0], list):
                vector = vector[0]
                
            return vector
            
    except Exception as e:
        print(f"❌ Error reading {image_path}: {e}")
        return None
# The pure math equation for Cosine Similarity
def calculate_cosine_similarity(v1, v2):
    dot_product = sum(a * b for a, b in zip(v1, v2))
    magnitude_v1 = math.sqrt(sum(a * a for a in v1))
    magnitude_v2 = math.sqrt(sum(b * b for b in v2))
    return dot_product / (magnitude_v1 * magnitude_v2)

# --- 1. Extract vectors for both images ---
vector_original = get_ai_vector('diddy.jpeg')
vector_stolen = get_ai_vector('diddy new nigga.jpeg')

if vector_original and vector_stolen:
    # --- 2. Calculate the angle between the two 512-dimensional arrays ---
    similarity = calculate_cosine_similarity(vector_original, vector_stolen)
    
    print("\n=== 🧬 AI VISUAL DNA MATCH RESULT ===")
    print(f"Cosine Similarity Score: {similarity * 100:.2f}%")
    
    if similarity > 0.90:
        print("🚨 VERDICT: STOLEN ASSET DETECTED (High Confidence)")
    elif similarity > 0.80:
        print("⚠️ VERDICT: SUSPICIOUS MODIFICATION (Review Needed)")
    else:
        print("✅ VERDICT: UNRELATED IMAGES")