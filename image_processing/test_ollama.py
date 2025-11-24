# test_ollama_simple.py
import sys
print(f"Python executable: {sys.executable}")
print(f"Python path: {sys.path[:3]}...")  # Show first 3 paths

try:
    import requests
    print("✓ Requests imported successfully")
    
    # Test Ollama API
    response = requests.get("http://localhost:11434/api/tags")
    print(f"✓ Ollama API responded with status: {response.status_code}")
    
    # Test generation
    response = requests.post("http://localhost:11434/api/generate", 
        json={
            "model": "gpt-oss",
            "prompt": "Say 'Hello, GPU!' in exactly 3 words.",
            "stream": False
        })
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Model response: {result['response']}")
    else:
        print(f"✗ Error: {response.status_code}")
        
except ImportError as e:
    print(f"✗ Import error: {e}")
except Exception as e:
    print(f"✗ Error: {e}")