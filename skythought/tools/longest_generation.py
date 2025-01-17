import numpy as np
import json

def main():
  dataset_path = "data/math5k/sky-t1-math5k-final.json"
  dataset = {}
  with open(dataset_path, 'r', encoding='utf-8') as file:
    dataset = json.load(file)

  gen_lens = []
  for prompt in dataset:
    problem = dataset[prompt]
    for usage in problem["token_usages"]:
      gen_lens.append(problem["token_usages"][usage]["completion_tokens"])

  print(np.percentile(gen_lens, 50))
  print(np.percentile(gen_lens, 75))
  print(np.percentile(gen_lens, 90))
  print(np.percentile(gen_lens, 98))
  print(np.percentile(gen_lens, 99))

if __name__ == "__main__":
    main()