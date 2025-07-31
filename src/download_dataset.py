import requests
import os

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
dataset_dir = "../data/raw"  # Directory to save the dataset
dataset_path = os.path.join(dataset_dir, "default_of_credit_card_clients.xls")  # Full path to the file

# Ensure the directory exists
os.makedirs(dataset_dir, exist_ok=True)

try:
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad responses (e.g., 404)
    with open(dataset_path, "wb") as file:
        file.write(response.content)
    print(f"Dataset downloaded successfully and saved as {dataset_path}!")
except requests.exceptions.RequestException as e:
    print(f"Failed to download the dataset: {e}")