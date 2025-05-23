import requests
import json

# 📍 IP адрес на HTB VM – увери се, че е коректен и машината е активна!
url = "http://10.129.220.99:5000/api/upload"

# 📦 Път до модела, който искаш да качиш
model_file_path = "skills_assessment.joblib"

# 📤 Качване
try:
    with open(model_file_path, "rb") as model_file:
        files = {"model": model_file}
        print(f"[*] Uploading model to {url} ...")
        response = requests.post(url, files=files, timeout=60)

        # ✅ Отговор от сървъра
        try:
            result = response.json()
            print(json.dumps(result, indent=4))
        except json.JSONDecodeError:
            print("[!] Server did not return valid JSON.")
            print(response.text)

except FileNotFoundError:
    print(f"[!] Model file not found: {model_file_path}")
except requests.exceptions.RequestException as e:
    print(f"[!] Request failed: {e}")
