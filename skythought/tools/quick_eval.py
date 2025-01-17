import json
from util.math.testing_util import strip_answer_string, get_multiple_choice_answer, extract_answer, math_equal

def main():
  dataset_path = "data/math5k/sky-t1-math5k-final.json"
  dataset = {}
  with open(dataset_path, 'r', encoding='utf-8') as file:
    dataset = json.load(file)

  scores = []
  for prompt in dataset:
    problem = dataset[prompt]
    response = problem["responses"]["trim"]
    gt_answer = problem["answer"]

    pred = extract_answer(response)
    pred = strip_answer_string(pred)
    scores.append(math_equal(pred, gt_answer))
  print("Score: ", sum(scores) / len(scores))

if __name__ == "__main__":
    main()