import requests
import json

# IP адрес на VM от Hack The Box
url = "http://10.129.231.90:5000/api/upload"

# Път до твоя модел (ако е в текущата директория)
model_file_path = "skills_assessment.joblib"

try:
    with open(model_file_path, "rb") as model_file:
        files = {"model": model_file}
        print(f"[*] Uploading model to {url} ...")
        response = requests.post(url, files=files, timeout=500)

        try:
            result = response.json()
            print("[✓] Server responded with JSON:")
            print(json.dumps(result, indent=4))
        except json.JSONDecodeError:
            print("[!] Server did not return valid JSON.")
            print(response.text)

except FileNotFoundError:
    print(f"[!] Model file not found: {model_file_path}")
except requests.exceptions.RequestException as e:
    print(f"[!] Request failed: {e}")
