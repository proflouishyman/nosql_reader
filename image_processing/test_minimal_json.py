# text_to_json.py
import requests
import json
import os

def text_to_json(text_content, model="gpt-oss"):
    """Convert text to structured JSON"""
    
    prompt = f"""Convert the following text into structured JSON format.
Extract ALL information and organize it logically.
Return ONLY valid JSON, no explanations or markdown formatting.

Text:
{text_content}
"""
    
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1  # Low temperature for consistent JSON
            }
        }
    )
    
    if response.status_code == 200:
        result = response.json()['response']
        
        # Clean up common formatting issues
        result = result.strip()
        if result.startswith("```json"):
            result = result[7:]
        if result.startswith("```"):
            result = result[3:]
        if result.endswith("```"):
            result = result[:-3]
        
        # Validate JSON
        try:
            parsed = json.loads(result.strip())
            return json.dumps(parsed, indent=2)
        except json.JSONDecodeError:
            return result  # Return raw if parsing fails
    else:
        return f"Error: {response.status_code}"

# Main execution
if __name__ == "__main__":
    # Read from extracted_text file
    input_file = "extracted_text.txt"
    
    with open(input_file, "r", encoding="utf-8") as f:
        text_content = f.read()
    
    print("Converting text to JSON...")
    json_output = text_to_json(text_content)
    
    # Save to JSON file
    output_file = "extracted_data.json"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(json_output)
    
    print(f"JSON saved to: {output_file}")
    print(f"\nJSON output:\n{json_output}")