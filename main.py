
import requests

def classify_sentiment(text):
    url = "http://localhost:8080/predict"
    payload = {
        "inputs": [
            {
                "name": "text",
                "shape": [1],
                "datatype": "BYTES",
                "data": [text]
            }
        ]
    }
    response = requests.post(url, json=payload)
    return response.json()

text = "The company's profits have increased significantly this quarter."
result = classify_sentiment(text)
print(result)


