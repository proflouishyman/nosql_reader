# stream_response.py
import requests
import json

def stream_ollama(prompt, model="gpt-oss"):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": True
        },
        stream=True
    )
    
    print("Response: ", end="", flush=True)
    for line in response.iter_lines():
        if line:
            chunk = json.loads(line)
            print(chunk['response'], end='', flush=True)
            if chunk.get('done', False):
                print("\n")
                break

# Test it
stream_ollama("Explain the concept of streaming responses in Ollama.")