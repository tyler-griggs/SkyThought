from datasets import load_dataset
import json
import random

def main():
  # Step 1: Load the dataset from Hugging Face
  # Replace 'dataset_name' with the desired dataset, e.g., 'imdb' or 'squad'
  dataset = load_dataset('BAAI/TACO', split='train')  # Load the 'train' split

  # Step 2: Convert dataset to a list of dictionaries
  data = [dict(record) for record in dataset]

  # Step 3: Dump the data into a JSON file
  output_file = "data/taco/taco25k-original.json"
  with open(output_file, 'w', encoding='utf-8') as f:
      json.dump(data, f, ensure_ascii=False, indent=4)

  # Randomly sample 5,000 data points
  sample_size = 5000
  random.seed(42)  # For reproducibility
  sampled_data = random.sample(data, sample_size)

  # Save the sampled data as a new JSON file
  output_file = "data/taco/taco5k-original.json"
  with open(output_file, "w", encoding="utf-8") as f:
      json.dump(sampled_data, f, ensure_ascii=False, indent=4)

  # file_path = "data/sky-t1-17k/sky-t1-17k-original.json"
  # with open(file_path, 'r', encoding='utf-8') as f:
  #     dataset = json.load(f)

  # new_dataset = {}
  # for problem in dataset:
  #   new_dataset[problem['conversations'][0]['value']] = {}
  #   new_dataset[problem['conversations'][0]['value']]["question"] = problem['conversations'][0]['value']
  #   new_dataset[problem['conversations'][0]['value']]["answer"] = problem
  # print(len(dataset[0]['conversations']))

if __name__ == "__main__":
    main()