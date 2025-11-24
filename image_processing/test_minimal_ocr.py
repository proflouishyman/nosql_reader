# ocr_extract.py
import requests
import base64
import json

def image_to_base64(image_path):
    """Convert image file to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_text_from_image(image_path, model="llama3.2-vision:11b"):
    """Extract text from image using vision model"""
    
    # Convert image to base64
    image_base64 = image_to_base64(image_path)
    
    # Create prompt for OCR
    prompt = "Extract ALL text from this image. Return only the text content, nothing else."
    
    # Make API request
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "images": [image_base64],
            "stream": False
        }
    )
    
    if response.status_code == 200:
        return response.json()['response']
    else:
        return f"Error: {response.status_code} - {response.text}"

# Example usage
if __name__ == "__main__":
    # Define your image path here
    image_path = "/data/lhyman6/nosql_project/nosql/archives/Paper_mini/RDApp-598632Jones002.jpg"  # Change this to your image
    
    print("Extracting text from image...")
    extracted_text = extract_text_from_image(image_path)
    print(f"Extracted text:\n{extracted_text}")
    
    # Save to file for next step
    with open("extracted_text.txt", "w") as f:
        f.write(extracted_text)