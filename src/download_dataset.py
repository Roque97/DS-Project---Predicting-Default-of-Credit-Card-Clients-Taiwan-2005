import requests
import os

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"

def download_dataset():
    """Download the dataset from the specified URL and save it to the local directory."""
    # Ensure the dataset directory is always relative to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(project_root, "data", "raw")
    dataset_path = os.path.join(dataset_dir, "default_of_credit_card_clients.xls")

    # Ensure the directory exists
    os.makedirs(dataset_dir, exist_ok=True)

    try:
        response = requests.get(url, verify=False)
        response.raise_for_status()  # Raise an error for bad responses (e.g., 404)
        with open(dataset_path, "wb") as file:
            file.write(response.content)
        print(f"Dataset downloaded successfully and saved as {dataset_path}!")

    except requests.exceptions.RequestException as e:
        print(f"Failed to download the dataset: {e}")

if __name__ == "__main__":
    download_dataset()