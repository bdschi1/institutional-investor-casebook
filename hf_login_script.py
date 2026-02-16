import os
from dotenv import load_dotenv
from huggingface_hub import login

# Load the token from your .env file
load_dotenv()
token = os.getenv("HF_TOKEN")

if token:
    print("Found token in .env. Attempting login...")
    login(token=token)
    print("Login successful! Token is now cached for this environment.")
else:
    print("ERROR: HF_TOKEN not found in .env file.")
