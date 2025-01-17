import json
from util.math.testing_util import strip_answer_string, get_multiple_choice_answer, extract_answer, math_equal

def main():
  answer_filepath = 'data/Sky-T1_math_5k.json'
  answer_dataset = {}
  with open(answer_filepath, 'r', encoding='utf-8') as file:
    answer_dataset = json.load(file)

  new_filepath = 'data/math5k/sky-t1-math5k-original.json'
  new_dataset = {}
  with open(new_filepath, 'r', encoding='utf-8') as file:
    new_dataset = json.load(file)

  for idx, problem in enumerate(new_dataset):
    solution = answer_dataset[idx]["conversations"][1]["value"]
    solution = extract_answer(solution)
    solution = strip_answer_string(solution)
    new_dataset[problem]["answer"] = solution

  new_filepath = 'data/math5k/sky-t1-math5k-original-answers.json'
  with open(new_filepath, 'w', encoding='utf-8') as new_file:
      json.dump(new_dataset, new_file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
