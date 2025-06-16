import requests

api_url = "http://localhost:8011/v1/chat/completions"

test_payload = {
    "model": "glm",
    "messages": [
        {"role": "system", "content": "你是由阿里云开发的通义千问大模型，能够回答各种问题。"},
        {"role": "user", "content": "你好，请介绍自己。"}
    ],
    # "prompt": "你是谁？",
    "temperature": 0.7,
    "max_tokens": 100,
    "top_k": -1
}

response = requests.post(api_url, json=test_payload)
print(response.json())