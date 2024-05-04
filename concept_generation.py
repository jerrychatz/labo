import argparse
import json
import os
import re
from pprint import pprint

import requests
import yaml
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


def save_json_data(data, filepath):
    with open(filepath, 'w') as json_file:
        json.dump(data, json_file, indent=4)
class OpenAIHelper:
    """A helper class for interacting with the OpenAI API."""
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("No OPENAI_API_KEY found in environment variables")
        self.client = OpenAI(api_key=api_key)

    def fetch_data(self, system_content, user_content):
        completion = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content}
            ]
        )
        return completion.choices[0].message.content


def load_yaml_config(filepath):
    try:
        with open(filepath, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print("YAML file not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error reading YAML file: {e}")
        return None


def save_yaml_data(data, filepath):
    with open(filepath, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

def fetch_data_from_openai_api_single_pass(config, system_content: str, user_content: str, client=None)->str:
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"{system_content}"},
            {"role": "user", "content": f"{user_content}"}
        ],
        max_tokens=20,
        )
    return completion.choices[0].message.content


def parse_arguments():
    parser = argparse.ArgumentParser(description="Openai API - Concept Generation Pipeline")
    parser.add_argument("--config_yaml", default="helper.yaml", help="Path to the YAML configuration file")
    parser.add_argument("--output_path", default="concepts.yaml", help="Path to save the processed YAML output")
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_arguments()


    # Setup Connection with OpenAI API Client
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    openai_helper = OpenAIHelper(api_key=api_key)

    config = load_yaml_config(args.config_yaml)
    if not config:
        return

    system_content = config['openai']['system-content']['cifar']
    prompt_templates = config['openai']['prompt-content-templates']
    cifar_classes = config['cifar10-classes']
    concepts_dict = {}

    for class_name in tqdm(cifar_classes):
        concepts_dict[class_name] = []
        for _ in tqdm(range(10)):
            for prompt_template in prompt_templates:
                prompt = prompt_template.replace("[CLASS NAME]", class_name)
                response = openai_helper.fetch_data('', prompt)
                concepts_dict[class_name].extend(response.split('. '))
        save_json_data(concepts_dict, args.output_path.replace('.yaml', '.json'))

if __name__ == "__main__":
    main()