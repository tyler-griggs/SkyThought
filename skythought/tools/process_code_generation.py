import json
from util.model_utils import *

# 998 in dataset
# 545 have correct
# 438 have at least 2 correct

def filter_solutions(scored_dataset, outfile):
  # First filter for correct solutions.
  for key in scored_dataset:
    problem = scored_dataset[key]
    keys_to_filter = []
    for response_key in problem["responses"]:
      if not problem["responses"][response_key]["correctness"]:
        keys_to_filter.append(response_key)
    for k in keys_to_filter:
      del problem["responses"][k]
      del problem["token_usages"][k]

  # Next, filter out samples with <2 correct solutions.
  keys_to_filter = []
  for key in scored_dataset:
    problem = scored_dataset[key]
    if len(problem["responses"]) < 2:
      keys_to_filter.append(key)
  for k in keys_to_filter:
    del scored_dataset[k]

  # Finally, filter for the shortest and longest solutions for each sample.
  for key in scored_dataset:
    problem = scored_dataset[key]
    token_usages = problem["token_usages"]
    shortest_key, shortest_entry = min(token_usages.items(), key=lambda x: x[1]["completion_tokens"])
    longest_key, longest_entry = max(token_usages.items(), key=lambda x: x[1]["completion_tokens"])

    problem["token_usages"] = {
      "shortest": shortest_entry,
      "longest": longest_entry,
    }

    new_responses = {
      "shortest": problem["responses"][shortest_key],
      "longest":problem["responses"][longest_key],
    }

    problem["responses"] = new_responses

  with open(outfile, 'w', encoding='utf-8') as new_file:
      json.dump(scored_dataset, new_file, ensure_ascii=False, indent=2)
  return scored_dataset


# # TODO(tgriggs): need to model this after TACO
# def format_to_simpo(final_dataset, outfile):
#   simpo_format = []
#   for prompt in final_dataset:
#     problem = final_dataset[prompt]
#     simpo = {}
#     simpo["conversations"] = [
#       {
#         "from": "system",
#         "value": SYSTEM_PROMPT["NovaSky-AI/Sky-T1-32B-Preview"]
#       },
#       {
#         "from": "human",
#         "value": "Return your final response within \\boxed{{}}" + prompt
#       }
#     ]
#     simpo["chosen"] = {
#       "from": "gpt",
#       "value": problem["responses"]["shortest"]["content"],
#     }
#     simpo["rejected"] = {
#       "from": "gpt",
#       "value": problem["responses"]["longest"]["content"]
#     }
#     simpo_format.append(simpo)

#   with open(outfile, 'w', encoding='utf-8') as new_file:
#       json.dump(simpo_format, new_file, ensure_ascii=False, indent=2)

def generate_prompt(prompt, starter_code=None, fn_name=None):
  _input = "Generate an executable Python function generated from the given prompt\nQUESTION:\n"
  _input += prompt
  if starter_code:
    _input += starter_code
  if (not fn_name) and (not starter_code):
    call_format = "\nUse Standard Input format"
    _input += call_format
  else:
    call_format = "\nUse Call-Based format"
    _input += call_format
  _input += "\nANSWER:\n"
  
  return _input

def make_conversations(dataset, outfile):
  conversations = []
  for prompt in dataset:
    problem = dataset[prompt]
    starter_code = None if len(problem["starter_code"]) == 0 else problem["starter_code"]
    try:
      input_outpout = json.loads(problem["input_output"])
      fn_name = (
          None if not input_outpout.get("fn_name") else input_outpout["fn_name"]
      )
    except ValueError:
      fn_name = None
    prompt_text = generate_prompt(problem["question"], starter_code, fn_name)
    
    simpo = {}
    simpo["conversations"] = [
      {
        "from": "system",
        "value": SYSTEM_PROMPT["NovaSky-AI/Sky-T1-32B-Preview"]
      },
      {
        "from": "human",
        "value": prompt_text
      }
    ]
    simpo["chosen"] = {
      "from": "gpt",
      "value": problem["responses"]["shortest"]["content"],
    }
    simpo["rejected"] = {
      "from": "gpt",
      "value": problem["responses"]["longest"]["content"]
    }
    conversations.append(simpo)
    
  with open(outfile, 'w', encoding='utf-8') as new_file:
    json.dump(conversations, new_file, ensure_ascii=False, indent=2)
    
  return conversations


def main():
  dataset_path = 'data/taco/Sky-T1-32B-Preview_TACO_train_None_0_-1.json'
  dataset = {}
  with open(dataset_path, 'r', encoding='utf-8') as file:
    dataset = json.load(file)

  scores = []
  for prompt in dataset:
    problem = dataset[prompt]
    score = 0
    for response_key in problem["responses"]:
      if problem["responses"][response_key]["correctness"]:
        score += 1
    scores.append(score)
    
  print("Dataset size: ", len(dataset))
  print("At least one correct response: ", len([x for x in scores if x >= 1]))
  print("At least two correct responses: ", len([x for x in scores if x >= 2]))
  
  filtered_dataset_path = "data/taco/sky-t1-preview-taco1k-filtered-solutions.json"
  filtered_dataset = filter_solutions(dataset, filtered_dataset_path)
  print("Filtered dataset length: ", len(filtered_dataset))
  
  # Make conversations
  conversation_dataset_path = "data/taco/sky-t1-preview-taco1k-conversations.json"
  conversation_dataset = make_conversations(dataset, conversation_dataset_path)
  print("Conversation dataset length: ", len(conversation_dataset))

if __name__ == "__main__":
    main()
