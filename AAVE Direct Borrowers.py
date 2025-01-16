import requests

url = "https://aave-api-v3.aave.com/borrowers"

response = requests.get(url)
if response.status_code == 200:
    data = response.json()
    print("Borrowers:", data)
else:
    print("Error fetching data:", response.status_code, response.text)
