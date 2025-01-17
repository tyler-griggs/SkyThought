import json


def get_total_tokens(dataset_path):
  dataset = {}
  with open(dataset_path, 'r', encoding='utf-8') as file:
    dataset = json.load(file)

  total_len = 0
  for prompt in dataset:
    problem = dataset[prompt]
    if problem["responses"]["0.7"]["correctness"]:
       total_len += problem["token_usages"]["0.7"]["completion_tokens"]

  return total_len

def main():
  tokens_before = get_total_tokens('results/Sky-T1-32B-Preview_MATH500_test_None_0_-1.json')
  tokens_after = get_total_tokens('results/Sky-T1-32B-Lightning-half_MATH500_test_None_0_-1.json')
  print(tokens_after / tokens_before)
       
if __name__ == "__main__":
    main()
