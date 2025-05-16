import requests

url = "http://localhost:8000/api/v1/predict"
payload = {"texts": ["I'm really disappointed. The service was terrible and slow.",'I hate this trend']}
response = requests.post(url, json=payload)

print("Status Code:", response.status_code)
print("Response:", response.json())