import requests
import json

# The URL of your local AI Brain
url = 'http://localhost:8000/analyze-asset'

# The image we want to test
image_path = 'diddy.jpeg' 

print(f"🚀 Sending {image_path} to the AI Engine...")

try:
    with open(image_path, 'rb') as file:
        # Package the file exactly how the API expects it
        files = {'file': file}
        response = requests.post(url, files=files)

    # Print the result beautifully!
    print("\n=== 🕵️‍♂️ AI FORENSIC REPORT ===")
    print(json.dumps(response.json(), indent=4))
    
except Exception as e:
    print(f"❌ Failed to test: {e}")