import mysql.connector
import requests

input_test = """Hi,
Kindly find the order details below:
Ordered on March 10, 2019
Order 113-5176542-9729865
Plantronics VOYAGER-5200-UC (206110-01) Advanced NC Bluetooth Headsets System
Sold by: TechFlash
Items shipped: March 11, 2019"""


response = requests.get("http://thing0.polycom.com:5000/segment", params={"input_text": input_test})

if response.status_code != 200:
    print("Failed with status code:", response.status_code)
sentiment = response.text
print(sentiment)