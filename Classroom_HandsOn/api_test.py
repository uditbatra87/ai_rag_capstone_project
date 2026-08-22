import requests
import os
from dotenv import load_dotenv

import warnings
warnings.filterwarnings("ignore")

# Load the variables
load_dotenv('.env')

# Access them using os.getenv or os.environ
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

url = 'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=IBM&apikey=ALPHA_VANTAGE_API_KEY'
r = requests.get(url)
data = r.json()

print(data)