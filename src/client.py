import requests
import time

# URL of the Flask endpoint serving the image
url = 'http://3.37.233.51:5001'
api_endpoint = '/genimage'


data = {
    "bot": "snow_white",
    "user": "witch",
    "summary": "cinematic photo casual elsa, <lora:add-detail-xl:1> <lora:princess_xl_v2:0.9>, . 35mm photograph, film, bokeh, professional, 4k, highly detailed, in the forest, sunny, summer, dress,smile"
}

start_time = time.time()

# Send a GET request to the Flask application
response = requests.post(url + api_endpoint, json=data)

# Record the end time
end_time = time.time()

# Calculate the time taken for the request
elapsed_time = end_time - start_time

# Check if the request was successful
if response.status_code == 200:
    # Open a file in binary write mode
    with open('images.zip', 'wb') as file:
        # Write the content of the response (which is the zip file) to a file
        file.write(response.content)
    print(f"Request took {elapsed_time:.2f} seconds to complete.")
else:
    print("Failed to retrieve the image. Status code:", response.status_code)
