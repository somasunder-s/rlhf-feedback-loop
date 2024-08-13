import json
import os
import pandas as pd


def load_csv(file_path):
    return pd.read_csv(file_path)


def save_json(data, file_name):
    directory = os.path.dirname(file_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)


def save_csv(data, file_name):
    data.to_csv(file_name, index=False)


def load_json(file_name):
    with open(file_name, 'r') as f:
        data = json.load(f)
    return data


def save_feedback_as_csv(json_list, filename):
    df = pd.DataFrame(json_list)
    df.to_csv(filename, index=False)
    print(f"DataFrame saved to {filename}")