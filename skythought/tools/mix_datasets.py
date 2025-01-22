import json
import random

# TACO
def main():
  dataset1 = {}
  dataset1_path = 'data/taco/sky-t1-preview-taco1k-conversations.json'
  with open(dataset1_path, 'r', encoding='utf-8') as file:
    dataset1 = json.load(file)
    
  dataset2 = {}
  dataset2_path = 'data/prm12k/sky-prm12k-simpo-fcs_reflection_trim.json'
  with open(dataset2_path, 'r', encoding='utf-8') as file:
    dataset2 = json.load(file)
    
  combined_dataset = dataset1 + dataset2

  random.shuffle(combined_dataset)

  output_path = 'data/mix/sky-prm12k-taco1k-conversations.json'
  with open(output_path, 'w', encoding='utf-8') as new_file:
    json.dump(combined_dataset, new_file, ensure_ascii=False, indent=2)

  print(f"Combined and shuffled dataset written to {output_path}")

# PRM12K
# def main():
#   dataset1 = {}
#   dataset1_path = 'data/prm12k/sky-prm12k-short-incorrect-conversations.json'
#   with open(dataset1_path, 'r', encoding='utf-8') as file:
#     dataset1 = json.load(file)
#   print(len(dataset1))
#   dataset1 = random.sample(dataset1, 450)
    
#   dataset2 = {}
#   dataset2_path = 'data/prm12k/sky-prm12k-simpo-fcs_reflection_trim.json'
#   with open(dataset2_path, 'r', encoding='utf-8') as file:
#     dataset2 = json.load(file)
#   print(len(dataset2))

#   combined_dataset = dataset1 + dataset2

#   random.shuffle(combined_dataset)
#   print(len(combined_dataset))

#   output_path = 'data/mix/sky-prm12k-mixed-SI450-conversations.json'
#   with open(output_path, 'w', encoding='utf-8') as new_file:
#     json.dump(combined_dataset, new_file, ensure_ascii=False, indent=2)

#   print(f"Combined and shuffled dataset written to {output_path}")


if __name__ == "__main__":
    main()
