import requests

# Define multiple XML payloads
xml_list = [
    "<xml><item>Item 1</item></xml>",
    "<xml><item>Item 2</item></xml>",
    "<xml><item>Item 3</item></xml>"
]

# Combine XMLs into one payload (e.g., wrap them in a root tag if required)
combined_xml = "<root>" + "".join(xml_list) + "</root>"

# Headers
headers = {
    "x-meta-feed-type": "1",
    "x-meta-feed-parameters": "feed params",
    "x-meta-default-filename": "filename.xml",
    "x-meta-game-id": "1",
    "x-meta-competition-id": "1",
    "x-meta-season-id": "2010",
    "x-meta-gamesystem-id": "1",
    "x-meta-matchday": "1",
    "x-meta-away-team-id": "1",
    "x-meta-home-team-id": "1",
    "x-meta-game-status": "11",
    "x-meta-language": "en",
    "x-meta-production-server": "server",
    "x-meta-production-server-timestamp": "1",
    "x-meta-production-server-module": "1",
    "x-meta-mime-type": "text/xml",
    "encoding": "UTF-8",
    "Content-Type": "application/xml"
}

# URL
url = 'http://localhost:8000/api/xml_secondary/'

# Send request
try:
    response = requests.post(url, data=combined_xml, headers=headers)
    if response.status_code == 200:
        print("Response Headers:")
        print(response.headers)
        print("\nResponse Body:")
        print(response.text)
    else:
        print(f"Error: Received status code {response.status_code}")
        print("Response Body:")
        print(response.text)
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
