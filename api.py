from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import torch
from PIL import Image, ImageEnhance, ImageFilter
import io
from transformers import CLIPProcessor, CLIPModel
from torchvision import transforms, models
import torch.nn as nn
import psycopg2
from psycopg2 import pool
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import hashlib
import json

# ==========================================
# CONFIGURATION
# ==========================================
SUPABASE_URI = "postgresql://postgres.gkksyrejaiqpxchcyvrb:UYLh%2Aq%24b7ayZMR%2B@aws-1-ap-south-1.pooler.supabase.com:5432/postgres"

# Connection pool for better database performance
connection_pool = None

# In-memory cache for vectors (LRU-style)
vector_cache = {}
CACHE_MAX_SIZE = 1000

# Analytics storage
analytics_data = {
    "total_requests": 0,
    "total_registrations": 0,
    "matches_found": 0,
    "avg_confidence": 0.0,
    "requests_by_hour": defaultdict(int),
    "top_matched_artists": defaultdict(int)
}

app = FastAPI(title="Art Protection API", version="2.0")

# ==========================================
# DATABASE CONNECTION POOL
# ==========================================
def init_db_pool():
    global connection_pool
    try:
        connection_pool = psycopg2.pool.SimpleConnectionPool(
            1, 10,
            SUPABASE_URI
        )
        print("Database connection pool initialized")
    except Exception as e:
        print(f"Warning: Could not initialize connection pool: {e}")

def get_db_connection():
    global connection_pool
    if connection_pool:
        try:
            return connection_pool.getconn()
        except:
            pass
    return psycopg2.connect(SUPABASE_URI)

def release_db_connection(conn):
    global connection_pool
    if connection_pool:
        try:
            connection_pool.putconn(conn)
        except:
            pass

# ==========================================
# MODEL LOADING
# ==========================================
clip_model_id = "openai/clip-vit-base-patch32"
clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
clip_model = CLIPModel.from_pretrained(clip_model_id)

device = torch.device("cpu")

stego_model = models.efficientnet_b0(pretrained=False)
stego_model.classifier[1] = nn.Linear(stego_model.classifier[1].in_features, 1)
stego_model.load_state_dict(torch.load('stego_model.pt', map_location=device))
stego_model.eval()

stego_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# IMAGE PREPROCESSING & AUGMENTATION
# ==========================================
def augment_image(image: Image.Image, augmentations: List[str]) -> List[Tuple[Image.Image, float]]:
    """Generate augmented versions of an image with weights."""
    augmented = [(image, 1.0)]  # Original image with full weight

    for aug_type in augmentations:
        if aug_type == "rotate_15":
            augmented.append((image.rotate(15, expand=True), 0.7))
        elif aug_type == "rotate_-15":
            augmented.append((image.rotate(-15, expand=True), 0.7))
        elif aug_type == "rotate_90":
            augmented.append((image.rotate(90, expand=True), 0.6))
        elif aug_type == "flip_h":
            augmented.append((image.transpose(Image.FLIP_LEFT_RIGHT), 0.8))
        elif aug_type == "flip_v":
            augmented.append((image.transpose(Image.FLIP_TOP_BOTTOM), 0.8))
        elif aug_type == "brightness_up":
            augmented.append((ImageEnhance.Brightness(image).enhance(1.2), 0.7))
        elif aug_type == "brightness_down":
            augmented.append((ImageEnhance.Brightness(image).enhance(0.8), 0.7))
        elif aug_type == "contrast_up":
            augmented.append((ImageEnhance.Contrast(image).enhance(1.2), 0.7))
        elif aug_type == "contrast_down":
            augmented.append((ImageEnhance.Contrast(image).enhance(0.8), 0.7))
        elif aug_type == "blur":
            augmented.append((image.filter(ImageFilter.GaussianBlur(1)), 0.5))

    return augmented

def get_clip_vector(image_bytes: bytes) -> List[float]:
    """Extract CLIP vector from image bytes."""
    cache_key = hashlib.md5(image_bytes).hexdigest()

    if cache_key in vector_cache:
        return vector_cache[cache_key]

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    dummy_text = [" "]
    text_inputs = clip_processor(text=dummy_text, return_tensors="pt", padding=True)
    inputs["input_ids"] = text_inputs["input_ids"]
    inputs["attention_mask"] = text_inputs["attention_mask"]

    with torch.no_grad():
        outputs = clip_model(**inputs)
        clip_vector = outputs.image_embeds[0].flatten().tolist()

    # Cache management
    if len(vector_cache) >= CACHE_MAX_SIZE:
        vector_cache.pop(next(iter(vector_cache)))
    vector_cache[cache_key] = clip_vector

    return clip_vector

def get_five_vectors(image_bytes: bytes) -> Dict[str, List[float]]:
    """Extract 5 vectors (whole + 4 quadrants) from image."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h = image.size

    quadrants = {
        "full_image": image,
        "top_left": image.crop((0, 0, w//2, h//2)),
        "top_right": image.crop((w//2, 0, w, h//2)),
        "bottom_left": image.crop((0, h//2, w//2, h)),
        "bottom_right": image.crop((w//2, h//2, w, h))
    }

    vectors = {}
    for name, crop in quadrants.items():
        img_byte_arr = io.BytesIO()
        crop.save(img_byte_arr, format='JPEG')
        crop_bytes = img_byte_arr.getvalue()
        vectors[name] = get_clip_vector(crop_bytes)

    return vectors

def get_multi_scale_vectors(image_bytes: bytes, scales: List[float] = [0.5, 0.75, 1.0]) -> Dict[str, List[float]]:
    """Extract vectors at multiple scales for robust matching."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    vectors = {}

    for scale in scales:
        if scale == 1.0:
            scaled_img = image
        else:
            new_size = (int(image.width * scale), int(image.height * scale))
            scaled_img = image.resize(new_size, Image.Resampling.LANCZOS)

        img_byte_arr = io.BytesIO()
        scaled_img.save(img_byte_arr, format='JPEG')
        scaled_bytes = img_byte_arr.getvalue()

        key = f"scale_{int(scale*100)}"
        vectors[key] = get_clip_vector(scaled_bytes)

    return vectors

def get_ensemble_vector(image_bytes: bytes, augmentations: List[str] = None) -> List[float]:
    """Get ensemble vector by averaging multiple augmented views."""
    if augmentations is None:
        augmentations = ["flip_h", "brightness_up", "contrast_up"]

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    augmented_images = augment_image(image, augmentations)

    all_vectors = []
    weights = []

    for aug_img, weight in augmented_images:
        img_byte_arr = io.BytesIO()
        aug_img.save(img_byte_arr, format='JPEG')
        aug_bytes = img_byte_arr.getvalue()
        vector = get_clip_vector(aug_bytes)
        all_vectors.append(np.array(vector))
        weights.append(weight)

    # Weighted average
    weighted_vectors = np.average(all_vectors, axis=0, weights=weights)
    return weighted_vectors.tolist()

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    v1 = np.array(vec1)
    v2 = np.array(vec2)
    similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return float(similarity)

# ==========================================
# ANALYTICS TRACKING
# ==========================================
def track_request(endpoint: str, match_found: bool = False, confidence: float = 0.0, artist: str = None):
    """Track analytics for API requests."""
    analytics_data["total_requests"] += 1
    hour = datetime.now().hour
    analytics_data["requests_by_hour"][hour] += 1

    if match_found:
        analytics_data["matches_found"] += 1
        if artist:
            analytics_data["top_matched_artists"][artist] += 1

    # Update running average confidence
    total = analytics_data["total_requests"]
    avg = analytics_data["avg_confidence"]
    analytics_data["avg_confidence"] = ((avg * (total - 1)) + confidence) / total

# ==========================================
# ENDPOINTS
# ==========================================
@app.post("/analyze-asset")
async def analyze_asset(file: UploadFile = File(...)):
    """Analyze an asset for steganography and get similarity vector."""
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Get ensemble vector for better accuracy
        ensemble_vector = get_ensemble_vector(image_bytes)

        # Run steganography detection
        stego_input = stego_transform(image).unsqueeze(0)
        with torch.no_grad():
            stego_output = stego_model(stego_input)
            probability = torch.sigmoid(stego_output).item()
            is_stego = probability > 0.85

        track_request("/analyze-asset", confidence=probability)

        return {
            "status": "success",
            "similarity_vector": ensemble_vector,
            "steganography_detected": is_stego,
            "stego_confidence": round(probability * 100, 2),
            "image_dimensions": {"width": image.width, "height": image.height},
            "analysis_method": "ensemble_augmentation"
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/local-db-search")
async def local_db_search(file: UploadFile = File(...)):
    """Search using 5 vectors (full + 4 quadrants), register matching region if duplicate found."""
    print(f"Searching database for {file.filename}...")

    image_bytes = await file.read()
    
    # Get 5 vectors (full + 4 quadrants)
    query_vectors = get_five_vectors(image_bytes)
    vector_string_map = {k: "[" + ",".join(map(str, v)) + "]" for k, v in query_vectors.items()}

    conn = get_db_connection()
    cursor = conn.cursor()

    best_match = None
    best_score = 0.0
    best_region = None

    try:
        # Search all 5 regions
        for region_name, vec_str in vector_string_map.items():
            cursor.execute("""
                SELECT artist_name, image_name, 1 - (visual_dna <=> %s) AS similarity
                FROM protected_assets 
                WHERE 1 - (visual_dna <=> %s) > 0.80
                ORDER BY similarity DESC
                LIMIT 1;
            """, (vec_str, vec_str))
            
            match = cursor.fetchone()
            if match and match[2] > best_score:
                best_score = match[2]
                best_match = match
                best_region = region_name

        # Decision tiers
        if best_match and best_score >= 0.95:
            tier = "DEFINITE_MATCH"
            action = "block_upload"
        elif best_match and best_score >= 0.85:
            tier = "PROBABLE_MATCH"
            action = "flag_for_review"
        elif best_match and best_score >= 0.80:
            tier = "LOW_CONFIDENCE"
            action = "proceed_to_web_search"
        else:
            tier = "NO_MATCH"
            action = "proceed_to_web_search"

        if best_match:
            print(f"Match: {best_match[0]} - {best_match[1]} ({best_score*100:.1f}%) in {best_region}")
            return {
                "match_found": tier != "NO_MATCH",
                "tier": tier,
                "action": action,
                "region": best_region,
                "evidence": {
                    "matched_artist": best_match[0],
                    "matched_file": best_match[1],
                    "confidence_score": round(best_score * 100, 2)
                }
            }
        else:
            # No match found - store full_image vector anyway
            print("No match, storing full_image vector")
            full_vec = query_vectors.get("full_image")
            if full_vec:
                vec_str = "[" + ",".join(map(str, full_vec)) + "]"
                cursor.execute("""
                    INSERT INTO protected_assets (artist_name, image_name, region, visual_dna, created_at)
                    VALUES (%s, %s, %s, %s, NOW())
                    RETURNING id;
                """, ("auto_register", file.filename, "full_image", vec_str))
                conn.commit()
            
            return {
                "match_found": False,
                "tier": "NO_MATCH",
                "action": "proceed_to_web_search",
                "evidence": None
            }

    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        cursor.close()
        release_db_connection(conn)

@app.post("/register-art")
async def register_art(
    artist_name: str = Form(...),
    file: UploadFile = File(...)
):
    """Register new artwork."""
    print(f"Registering new art for {artist_name}: {file.filename}...")

    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        conn = get_db_connection()
        cursor = conn.cursor()

        full_vector = get_clip_vector(image_bytes)
        vector_string = "[" + ",".join(map(str, full_vector)) + "]"

        # Check for duplicate
        cursor.execute("""
            SELECT artist_name, image_name, 1 - (visual_dna <=> %s) AS similarity
            FROM protected_assets 
            WHERE region = 'full_image'
            AND 1 - (visual_dna <=> %s) > 0.98
            LIMIT 1;
        """, (vector_string, vector_string))
        duplicate = cursor.fetchone()
        if duplicate:
            cursor.close()
            release_db_connection(conn)
            return {
                "status": "error",
                "message": "Duplicate image already registered",
                "duplicate_of": duplicate[0],
                "similarity": round(duplicate[1] * 100, 2)
            }

        # Save single vector
        cursor.execute("""
            INSERT INTO protected_assets (artist_name, image_name, region, visual_dna, created_at)
            VALUES (%s, %s, %s, %s, NOW())
            RETURNING id;
        """, (artist_name, file.filename, "full_image", vector_string))

        asset_id = cursor.fetchone()[0]
        conn.commit()
        cursor.close()
        release_db_connection(conn)

        print(f"Registered {file.filename} for {artist_name}!")
        return {
            "status": "success",
            "message": f"Secured {file.filename} for {artist_name}.",
            "asset_id": asset_id
        }

    except Exception as e:
        print(f"Error: {e}")
        return {"status": "error", "message": str(e)}
    finally:
        cursor.close()
        release_db_connection(conn)

@app.post("/register-art")
async def register_art(
    artist_name: str = Form(...),
    file: UploadFile = File(...),
    include_augmented: bool = Form(default=False),
    metadata: Optional[str] = Form(default=None)
):
    """
    Register new artwork with enhanced options.

    Args:
        artist_name: Name of the artist
        file: Image file to register
        include_augmented: Also store augmented versions for better matching
        metadata: Optional JSON metadata string
    """
    print(f"📥 Registering new art for {artist_name}: {file.filename}...")

    try:
        image_bytes = await file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        conn = get_db_connection()
        cursor = conn.cursor()

        # Use single full image vector only
        full_vector = get_clip_vector(image_bytes)
        vector_string = "[" + ",".join(map(str, full_vector)) + "]"

        # Check for exact duplicate
        cursor.execute("""
            SELECT artist_name, image_name, 1 - (visual_dna <=> %s) AS similarity
            FROM protected_assets 
            WHERE region = 'full_image'
            AND 1 - (visual_dna <=> %s) > 0.98
            ORDER BY similarity DESC
            LIMIT 1;
        """, (vector_string, vector_string))
        duplicate = cursor.fetchone()
        if duplicate:
            cursor.close()
            release_db_connection(conn)
            return {
                "status": "error",
                "message": "Duplicate image already registered",
                "duplicate_of": duplicate[0],
                "similarity": round(duplicate[1] * 100, 2)
            }

        # Save single vector
        cursor.execute("""
            INSERT INTO protected_assets (artist_name, image_name, region, visual_dna, created_at)
            VALUES (%s, %s, %s, %s, NOW())
            RETURNING id;
        """, (artist_name, file.filename, "full_image", vector_string))

        asset_id = cursor.fetchone()[0]
        regions_saved = [f"full_image(id:{asset_id})"]

        # Save metadata if provided
        if metadata:
            try:
                meta_dict = json.loads(metadata)
                for key, value in meta_dict.items():
                    cursor.execute("""
                        INSERT INTO asset_metadata (asset_filename, metadata_key, metadata_value)
                        VALUES (%s, %s, %s)
                    """, (file.filename, key, str(value)))
            except (json.JSONDecodeError, TypeError) as e:
                print(f"Warning: Invalid metadata JSON, skipping: {e}")

        conn.commit()
        cursor.close()
        release_db_connection(conn)

        analytics_data["total_registrations"] += 1

        print(f"✅ Successfully registered {file.filename} for {artist_name}!")
        return {
            "status": "success",
            "message": f"Secured {file.filename} for {artist_name}.",
            "asset_id": asset_id
        }

    except Exception as e:
        print(f"❌ Error: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/batch-analyze")
async def batch_analyze(files: List[UploadFile] = File(...)):
    """Analyze multiple images at once."""
    results = []

    for file in files:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        ensemble_vector = get_ensemble_vector(image_bytes)

        stego_input = stego_transform(image).unsqueeze(0)
        with torch.no_grad():
            stego_output = stego_model(stego_input)
            probability = torch.sigmoid(stego_output).item()

        results.append({
            "filename": file.filename,
            "steganography_detected": probability > 0.85,
            "stego_confidence": round(probability * 100, 2),
            "dimensions": {"width": image.width, "height": image.height}
        })

    return {
        "status": "success",
        "total_analyzed": len(results),
        "results": results
    }

@app.post("/bulk-register")
async def bulk_register(
    artist_name: str = Form(...),
    files: List[UploadFile] = File(...),
    include_augmented: bool = Form(default=False)
):
    """Register multiple artworks for an artist at once."""
    registered = []
    failed = []

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        for file in files:
            try:
                image_bytes = await file.read()
                img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                vectors = get_five_vectors(image_bytes)

                for region_name, vector in vectors.items():
                    vector_string = "[" + ",".join(map(str, vector)) + "]"
                    cursor.execute("""
                        INSERT INTO protected_assets (artist_name, image_name, region, visual_dna, created_at)
                        VALUES (%s, %s, %s, %s, NOW())
                    """, (artist_name, file.filename, region_name, vector_string))

                registered.append(file.filename)

            except Exception as e:
                failed.append({"filename": file.filename, "error": str(e)})

        conn.commit()

    finally:
        cursor.close()
        release_db_connection(conn)

    analytics_data["total_registrations"] += len(registered)

    return {
        "status": "success",
        "registered": registered,
        "failed": failed,
        "total_success": len(registered),
        "total_failed": len(failed)
    }

@app.get("/db/stats")
async def get_db_stats():
    """Get database statistics and analytics."""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Total assets
        cursor.execute("SELECT COUNT(*) FROM protected_assets")
        total_assets = cursor.fetchone()[0]

        # Assets by artist
        cursor.execute("""
            SELECT artist_name, COUNT(*) as count
            FROM protected_assets
            GROUP BY artist_name
            ORDER BY count DESC
            LIMIT 10
        """)
        artists = cursor.fetchall()

        # Assets by region
        cursor.execute("""
            SELECT region, COUNT(*) as count
            FROM protected_assets
            GROUP BY region
            ORDER BY count DESC
        """)
        regions = cursor.fetchall()

        return {
            "status": "success",
            "total_assets": total_assets,
            "by_artist": [{"artist": r[0], "count": r[1]} for r in artists],
            "by_region": [{"region": r[0], "count": r[1]} for r in regions]
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        cursor.close()
        release_db_connection(conn)

@app.get("/db/cleanup")
async def cleanup_database(days_old: int = 30):
    """
    Cleanup old or duplicate entries.

    Args:
        days_old: Remove entries older than this many days (default 30)
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Find duplicates (same artist, image, region)
        cursor.execute("""
            SELECT artist_name, image_name, region, COUNT(*), MIN(id), MAX(id)
            FROM protected_assets
            GROUP BY artist_name, image_name, region
            HAVING COUNT(*) > 1
        """)
        duplicates = cursor.fetchall()

        deleted_count = 0
        for dup in duplicates:
            # Keep the oldest, delete newer duplicates
            cursor.execute("""
                DELETE FROM protected_assets
                WHERE artist_name = %s AND image_name = %s AND region = %s AND id != %s
            """, (dup[0], dup[1], dup[2], dup[4]))
            deleted_count += cursor.rowcount

        # Remove old entries if requested
        old_count = 0
        if days_old:
            cursor.execute("""
                DELETE FROM protected_assets
                WHERE created_at < NOW() - INTERVAL '%s days'
            """, (days_old,))
            old_count = cursor.rowcount

        conn.commit()

        return {
            "status": "success",
            "duplicates_removed": deleted_count,
            "old_entries_removed": old_count,
            "total_cleaned": deleted_count + old_count
        }

    finally:
        cursor.close()
        release_db_connection(conn)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    health = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "clip": clip_model is not None,
            "stego": stego_model is not None
        },
        "cache_size": len(vector_cache),
        "cache_max": CACHE_MAX_SIZE
    }

    # Check database connection
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.close()
        release_db_connection(conn)
        health["database"] = "connected"
    except Exception as e:
        health["database"] = f"error: {str(e)}"
        health["status"] = "degraded"

    return health

@app.get("/analytics")
async def get_analytics():
    """Get detailed API analytics."""
    return {
        "overview": {
            "total_requests": analytics_data["total_requests"],
            "total_registrations": analytics_data["total_registrations"],
            "total_matches": analytics_data["matches_found"],
            "match_rate": round(analytics_data["matches_found"] / max(analytics_data["total_requests"], 1) * 100, 2),
            "avg_confidence": round(analytics_data["avg_confidence"] * 100, 2)
        },
        "hourly_distribution": dict(analytics_data["requests_by_hour"]),
        "top_matched_artists": dict(sorted(
            analytics_data["top_matched_artists"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10])
    }

@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    print("=" * 50)
    print("🚀 Dual-Engine Python API v2.0 Starting...")
    print("=" * 50)
    init_db_pool()
    print("✅ All systems ready!")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
