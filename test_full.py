import subprocess
import sys
import time
import requests

# Start server in background
print("Starting API server...")
proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"],
    cwd=".",
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

# Wait for server to start
time.sleep(15)

API_URL = "http://localhost:8000"

try:
    # Test API docs
    resp = requests.get(f"{API_URL}/docs", timeout=10)
    print(f"API Docs Status: {resp.status_code}")
    
    # Test register endpoint
    with open("diddy.jpeg", "rb") as f:
        files = {"file": f}
        data = {"artist_name": "Test Artist"}
        resp = requests.post(f"{API_URL}/register-art", files=files, data=data, timeout=120)
        print(f"/register-art Status: {resp.status_code}")
        print(f"Response: {resp.text}")
except Exception as e:
    print(f"Error: {e}")
finally:
    proc.terminate()
    proc.wait()