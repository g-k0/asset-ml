import requests
import psycopg2

API_URL = "http://localhost:8000"

# Test 1: Check if API is running
try:
    resp = requests.get(f"{API_URL}/docs")
    print(f"API Status: {resp.status_code}")
except Exception as e:
    print(f"API not running: {e}")
    exit(1)

# Test 2: Try to connect to Supabase directly (with URL-encoded password)
SUPABASE_URI = "postgresql://postgres.gkksyrejaiqpxchcyvrb:UYLh%2Aq%24b7ayZMR%2B@aws-1-ap-south-1.pooler.supabase.com:5432/postgres"
try:
    conn = psycopg2.connect(SUPABASE_URI)
    print("Supabase: Connected!")
    conn.close()
except Exception as e:
    print(f"Supabase Connection Error: {e}")

# Test 3: Test /register-art endpoint
IMAGE_PATH = "diddy.jpeg"
try:
    with open(IMAGE_PATH, 'rb') as f:
        files = {'file': f}
        data = {'artist_name': 'Test Artist'}
        resp = requests.post(f"{API_URL}/register-art", files=files, data=data, timeout=60)
        print(f"/register-art Status: {resp.status_code}")
        print(f"Response: {resp.text}")
except Exception as e:
    print(f"Register Error: {e}")