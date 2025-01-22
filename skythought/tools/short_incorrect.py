import json
import random

def query_for_stats(dataset):
  # First, how many questions have at least one incorrect answer?
  # Second, of those incorrect answers, how many have a correct answer longer than it?

  at_least_one_wrong = 0
  at_least_one_wrong_and_long = 0
  for prompt in dataset:
    problem = dataset[prompt]
    for response_key in problem["responses"]:
       if not problem["responses"][response_key]['correctness']:
          at_least_one_wrong += 1
          wrong_length = problem["token_usages"][response_key]['completion_tokens']

          for k in problem["responses"]:
            if k != response_key and problem["token_usages"][k]['completion_tokens'] > wrong_length and problem["responses"][k]['correctness']:
              at_least_one_wrong_and_long +=1
              break

          break
  print(at_least_one_wrong, " / ", len(dataset))
  print(at_least_one_wrong_and_long, " / ", len(dataset))


def main():
  # dataset_path = 'data/math5k/sky-t1-math5k-scored-solutions.json'
  dataset_path = 'data/prm12k/sky-prm12k-scored-solutions.json'
  dataset = {}
  with open(dataset_path, 'r', encoding='utf-8') as file:
    dataset = json.load(file)

  # query_for_stats(dataset)

  # Filter out all questions without incorrect answer, or correct answer that's longer
  keys_to_filter = []
  for prompt in dataset:
    problem = dataset[prompt]
    at_least_one_wrong = False
    at_least_one_wrong_and_long = False
    for response_key in problem["responses"]:
       if not problem["responses"][response_key]['correctness'] and not problem["responses"][response_key]['content'].endswith("I need to"):
          at_least_one_wrong = True
          wrong_length = problem["token_usages"][response_key]['completion_tokens']
          for k in problem["responses"]:
            if k != response_key and problem["token_usages"][k]['completion_tokens'] > wrong_length and problem["responses"][k]['correctness']:
              at_least_one_wrong_and_long = True
              break
          break
    if not at_least_one_wrong or not at_least_one_wrong_and_long:
      keys_to_filter.append(prompt)

  for key in keys_to_filter:
    del dataset[key]
  print('Len after filtering: ', len(dataset))

  outfile = 'data/prm12k/sky-prm12k-short-incorrect-filter.json'
  with open(outfile, 'w', encoding='utf-8') as new_file:
      json.dump(dataset, new_file, ensure_ascii=False, indent=2)


  # Next build contrastive pairs out of {short incorrect, long correct}
  simpo_format = []
  for prompt in dataset:
    problem = dataset[prompt]

    shortest_incorrect_key = None
    shortest_incorrect_length = float('inf')

    for response_key in problem["responses"]:
      if problem["responses"][response_key]['correctness'] is False and not problem["responses"][response_key]['content'].endswith("I need to"):
        length = problem["token_usages"][response_key]['completion_tokens']
        if length < shortest_incorrect_length:
          shortest_incorrect_length = length
          shortest_incorrect_key = response_key

    shortest_correct_longer_key = None
    shortest_correct_longer_length = float('inf')
    for response_key in problem["responses"]:
      if problem["responses"][response_key]['correctness'] is True:
        length = problem["token_usages"][response_key]['completion_tokens']
        if length > shortest_incorrect_length and length < shortest_correct_longer_length:
          shortest_correct_longer_length = length
          shortest_correct_longer_key = response_key

    simpo = {}
    simpo["conversations"] = [
      {
        "from": "system",
        "value": "Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format: <|begin_of_thought|> {thought with steps separated with '\n\n'} <|end_of_thought|> Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <|begin_of_solution|> {final formatted, precise, and clear solution} <|end_of_solution|> Now, try to solve the following question through the above guidelines:",
      },
      {
        "from": "human",
        "value": "Return your final response within \\boxed{{}}. " + prompt
      }
    ]
    simpo["chosen"] = {
      "from": "gpt",
      "value": problem["responses"][shortest_correct_longer_key]['content'],
    }
    simpo["rejected"] = {
      "from": "gpt",
      "value": problem["responses"][shortest_incorrect_key]["content"]
    }
    simpo_format.append(simpo)

  outfile = 'data/prm12k/sky-prm12k-short-incorrect-conversations.json'
  with open(outfile, 'w', encoding='utf-8') as new_file:
      json.dump(simpo_format, new_file, ensure_ascii=False, indent=2)

  # Now add it to the existing PRM12K dataset and shuffle
  full_dataset_path = 'data/prm12k/sky-prm12k-simpo-fcs_reflection_trim.json'
  with open(full_dataset_path, 'r', encoding='utf-8') as file:
    full_dataset = json.load(file)
  print('Len of original: ', len(full_dataset))

  combined_dataset = full_dataset + simpo_format
  random.shuffle(combined_dataset)
  print('Len after combining: ', len(combined_dataset))

  outfile = 'data/prm12k/sky-prm12k-mixed-short-incorrect-conversations.json'
  with open(outfile, 'w', encoding='utf-8') as new_file:
      json.dump(combined_dataset, new_file, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
