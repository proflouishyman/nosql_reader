import os
import sys
print(f"Python: {sys.version}")
print(f"Environment proxies: {[k for k in os.environ if 'proxy' in k.lower()]}")

try:
    from openai import OpenAI
    print(f"OpenAI imported successfully")
    client = OpenAI(api_key="sk-test")
    print("Client created successfully!")
except Exception as e:
    import traceback
    print(f"Error: {e}")
    print(traceback.format_exc())