# tests/test_api.py

import requests

def test_allocate_api():
    url = "http://127.0.0.1:8000/api/allocate"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        print("API 返回数据：", data)
    else:
        print("请求失败，状态码：", response.status_code)

if __name__ == "__main__":
    test_allocate_api()

