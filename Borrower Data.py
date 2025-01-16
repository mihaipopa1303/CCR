import requests

# Define GraphQL endpoint and query
url = "https://api.thegraph.com/subgraphs/name/aave/protocol-v2"
query = """
{
  users(first: 10, where: {borrowedAmount_gt: "0"}) {
    id
    borrowedAmount
    collateral
  }
}
"""

try:
    # Send request
    response = requests.post(url, json={'query': query})
    response.raise_for_status()  # Raise an error for HTTP issues

    # Debug response
    print("HTTP Status Code:", response.status_code)
    print("Response Content:", response.text)

    # Parse JSON
    json_response = response.json()
    if 'errors' in json_response:
        print("GraphQL Query Error:", json_response['errors'])
        exit()

    # Extract data
    data = json_response.get('data', {}).get('users', [])
    if not data:
        print("No user data found.")
    else:
        print("User Data:", data)

except requests.exceptions.RequestException as e:
    print("HTTP Request Error:", e)
except KeyError as e:
    print("Parsing Error:", e)
