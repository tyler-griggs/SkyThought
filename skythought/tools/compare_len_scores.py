import json


def get_total_tokens(dataset_path):
  dataset = {}
  with open(dataset_path, 'r', encoding='utf-8') as file:
    dataset = json.load(file)

  len_by_difficulty = [0, 0, 0, 0, 0]
  scores_by_difficulty = [[], [], [], [], []]
  count_by_difficulty = [0, 0, 0, 0, 0]
  for prompt in dataset:
    problem = dataset[prompt]
    scores_by_difficulty[problem["level"]-1].append(problem["responses"]["0.7"]["correctness"])
    if problem["responses"]["0.7"]["correctness"]:
       len_by_difficulty[problem["level"]-1] += problem["token_usages"]["0.7"]["completion_tokens"]
       count_by_difficulty[problem["level"]-1] += 1


  return len_by_difficulty, scores_by_difficulty, count_by_difficulty

def main():
  before_filepath = '/home/ubuntu/tgriggs/test/SkyThought/skythought/tools/results/Sky-T1-32B-Preview_MATH500_test_None_0_-1.json'
  # after_filepath = '/home/ubuntu/tgriggs/test/SkyThought/skythought/tools/results/Sky-T1-32B-Lightning_MATH500_test_None_0_-1.json'
  after_filepath = '/home/ubuntu/tgriggs/test/SkyThought/skythought/tools/results/prm12k-2/Sky-T1-prm12k-fcs-reflection-2_MATH500_test_None_0_-1.json'
  len_by_difficulty_before, scores_by_difficulty_before, count_by_difficulty_before = get_total_tokens(before_filepath)
  len_by_difficulty_after, scores_by_difficulty_after, count_by_difficulty_after = get_total_tokens(after_filepath)
  print("Avg lengths by difficulty")
  avg_len_by_difficulty_before = [len_by_difficulty_before[i] / count_by_difficulty_before[i] for i in range(len(len_by_difficulty_before))]
  avg_len_by_difficulty_after = [len_by_difficulty_after[i] / count_by_difficulty_after[i] for i in range(len(len_by_difficulty_after))]
  print("Before: ", avg_len_by_difficulty_before)
  print("After: ", avg_len_by_difficulty_after)
  for i in range(len(avg_len_by_difficulty_before)):
    print("Level ", i + 1, ": ", avg_len_by_difficulty_after[i] / avg_len_by_difficulty_before[i])
  print("Total: ", sum(avg_len_by_difficulty_after) / sum(avg_len_by_difficulty_before))
  print("Overall avg len: ", sum(len_by_difficulty_after) / sum(count_by_difficulty_after))

  print("\n\n")
  print("Scores by difficulty")
  for i in range(len(scores_by_difficulty_before)):
    score_before = sum(scores_by_difficulty_before[i]) / len(scores_by_difficulty_before[i])
    score_after = sum(scores_by_difficulty_after[i]) / len(scores_by_difficulty_after[i])
    print("Level ", i + 1, ": ", score_before, "(before), ", score_after, " (after), ", score_after - score_before, "(diff)")

if __name__ == "__main__":
    main()
