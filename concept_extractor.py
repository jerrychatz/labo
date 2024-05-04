import argparse
import json

import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer


def parse_arguments():
    parser = argparse.ArgumentParser(description="OpenAI API - Concept Generation Pipeline")
    parser.add_argument("--concepts-fp", default="concepts.json", help="Path to the concepts JSON file")
    parser.add_argument("--output-fp", default="concepts_t5_extracted.json", help="Path to output JSON file")
    return parser.parse_args()

def load_json(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data

def save_json(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

def generate_concepts(model, tokenizer, data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    generated_data = {}
    for key, descriptions in tqdm(data.items()):
        generated_concepts = []
        for description in tqdm(descriptions):
            inputs = tokenizer.encode("extract concepts from sentence: : " + description, return_tensors="pt")
            inputs = inputs.to(device)
            outputs = model.generate(inputs)
            generated_description = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_concepts.append(generated_description)
        generated_data[key] = generated_concepts
        save_json(generated_data, args.output_fp)
    return generated_data


def main(args):
    # Load data
    data = load_json(args.concepts_fp)
    # Load model
    # Load the T5 model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-large")
    model = T5ForConditionalGeneration.from_pretrained("t5-large-concept-extractor")
    model.to('cuda')

    # Generate concepts
    generated_concepts = generate_concepts(model, tokenizer, data)

    # Save output
    save_json(generated_concepts, args.output_fp)

if __name__ == "__main__":
    args = parse_arguments()
    main(args)