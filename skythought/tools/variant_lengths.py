import json

def get_total_tokens(dataset, variant):
  total_len = 0
  for prompt in dataset:
    problem = dataset[prompt]
    total_len += problem["token_usages"][variant]["completion_tokens"]

  return total_len

def main():
  dataset_path = 'data/math5k/sky-t1-math5k-final.json'

  dataset = {}
  with open(dataset_path, 'r', encoding='utf-8') as file:
    dataset = json.load(file)

  print("Shortest: ", get_total_tokens(dataset, 'shortest'))
  print("fcs: ", get_total_tokens(dataset, 'fcs_trim'), get_total_tokens(dataset, 'fcs_trim') / get_total_tokens(dataset, 'shortest'))
  print("fcs_reflection: ", get_total_tokens(dataset, 'fcs_reflection_trim'), get_total_tokens(dataset, 'fcs_reflection_trim') / get_total_tokens(dataset, 'shortest'))
  # print("trim: ", get_total_tokens(dataset, 'trim'))
  print("Longest: ", get_total_tokens(dataset, 'longest'), get_total_tokens(dataset, 'longest') / get_total_tokens(dataset, 'shortest'))

  
       
if __name__ == "__main__":
    main()
