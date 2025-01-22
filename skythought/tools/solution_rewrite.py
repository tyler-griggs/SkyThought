import argparse
import json
from vllm import LLM, SamplingParams
from util.math.testing_util import strip_answer_string, get_multiple_choice_answer, extract_answer, math_equal
from tqdm import tqdm
from util.model_utils import *


# TODO
  # DONE First run answer_extractor
  # DONE Then make a scoring run
  # DONE Then run full filtering run
  # DONE Then run real quick_eval on 'shortest' generation to confirm it's actually correct
  # Change FCS and FCS+Reflection code to not include "False" sections if they don't lead to an answer?

# TODO
  # Debug the weird occurences where there's a mismatch between whether a subsolution is correct and if the full solution is correct
  # Collect statistics on correctness over solutions

def load_dataset(dataset_path : str):
  # Open and read the JSONL file
  data = {}
  with open(dataset_path, 'r', encoding='utf-8') as file:
    data = json.load(file)
  return data

def make_scoring_conversations(dataset, system_prompt):
  conversations = []
  for idx, key in enumerate(dataset):
    problem = dataset[key]
    gt_answer = strip_answer_string(problem["answer"])
    for response_key in problem["responses"]:
      response = problem["responses"][response_key]["content"]
      prompt_text = response + "\n#####\nThe ground truth answer is " + gt_answer
      conversations.append([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_text}
      ])

  return conversations

def score_solutions(dataset, responses, outfile):
  idx = 0
  for dataset_idx, key in tqdm(enumerate(dataset), total=len(dataset), desc="Scoring original solutions"):
    problem = dataset[key]
    for response_key in problem["responses"]:
      score = responses[idx].outputs[0].text.strip()
      problem["responses"][response_key]["correctness"] = (score == "True")
      idx += 1

  with open(outfile, 'w', encoding='utf-8') as new_file:
      json.dump(dataset, new_file, ensure_ascii=False, indent=2)
  return dataset

def simple_score_solutions(dataset, outfile):
  for _, key in tqdm(enumerate(dataset), total=len(dataset), desc="Scoring original solutions"):
    problem = dataset[key]
    gt_answer = problem["answer"]
    for response_key in problem["responses"]:
      pred = problem["responses"][response_key]["content"]

      pred = extract_answer(pred)
      pred = strip_answer_string(pred)
      score = math_equal(pred, gt_answer)
      problem["responses"][response_key]["correctness"] = score

  with open(outfile, 'w', encoding='utf-8') as new_file:
      json.dump(dataset, new_file, ensure_ascii=False, indent=2)
  return dataset

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


def make_splitting_conversations(data, system_prompt):
  conversations = []
  for problem in data:
    response = data[problem]["responses"]["shortest"]
    prompt_text = response["content"]
    conversations.append([
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": prompt_text}
    ])
  return conversations

def split_solutions(responses):
  outputs = []
  for idx, response in tqdm(enumerate(responses), total=len(responses), desc="Splitting responses"):
    content = response.outputs[0].text.strip()

    # Split response by separator string
    split_content = content.split('#####')
    split_content = [x.strip() for x in split_content if x != ""]

    outputs.append(split_content)
  return outputs
    
def add_split_to_dataset(dataset, subsolutions, outfile):
  for idx, key in enumerate(dataset):
    solutions = subsolutions[idx]
    problem = dataset[key]
    problem["responses"]["shortest"]["subsolutions"] = solutions

  with open(outfile, 'w', encoding='utf-8') as new_file:
      json.dump(dataset, new_file, ensure_ascii=False, indent=2)
  return dataset

def make_subscoring_conversations(dataset, system_prompt):
  conversations = []
  for idx, key in enumerate(dataset):
    problem = dataset[key]
    gt_answer = strip_answer_string(problem["answer"])
    subsolutions = problem["responses"]["shortest"]["subsolutions"]
    for sub in subsolutions:
      prompt_text = sub + "\n#####\nThe ground truth answer is " + gt_answer
      conversations.append([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_text}
      ])

  return conversations

def score_subsolutions(dataset, responses, outfile):
  idx = 0
  for dataset_idx, key in tqdm(enumerate(dataset), total=len(dataset), desc="Scoring sub-solutions"):
    problem = dataset[key]
    subsolutions = problem["responses"]["shortest"]["subsolutions"]
    scores = []
    for _, sub in enumerate(subsolutions):
      score = responses[idx].outputs[0].text.strip()
      scores.append(score == "True")
      idx += 1
    problem["responses"]["shortest"]["scores"] = scores

  with open(outfile, 'w', encoding='utf-8') as new_file:
      json.dump(dataset, new_file, ensure_ascii=False, indent=2)
  return dataset


def build_solution_variants(dataset, outfile):

  def clean_response_string(response, gt_answer):
    if '<|end_of_thought|>' not in response:
      response += '<|end_of_thought|>'
    # if '<|begin_of_solution|>' not in response:
    #   response += '\n<|begin_of_solution|>\\boxed{{' + gt_answer + '}}<|end_of_solution|>'
    return response

  keys_to_filter = []
  for _, key in enumerate(dataset):
    problem = dataset[key]
    scores = problem["responses"]["shortest"]["scores"]
    subsolutions = problem["responses"]["shortest"]["subsolutions"]

    if True not in scores:
      keys_to_filter.append(key)
      continue

    # Trimmed FCS
    fcs_idx = scores.index(True)
    if fcs_idx == len(scores) - 1:
      fcs_response = "\n".join(subsolutions[:-1])
    else:
      fcs_response = "\n".join(subsolutions[:fcs_idx+1])

    fcs_response = clean_response_string(fcs_response, problem["answer"])
    fcs_response += "\n" + subsolutions[-1]
    problem["responses"]["fcs_trim"] = fcs_response

    # Trimmed FCS+Reflection
    if True not in scores[fcs_idx + 1:]:
      keys_to_filter.append(key)
      continue
    fcs_reflection_idx = scores.index(True, fcs_idx + 1)
    if fcs_reflection_idx == len(scores) - 1:
      fcs_reflection_response = "\n".join(subsolutions[:-1])
    else:
      fcs_reflection_response = "\n".join(subsolutions[:fcs_reflection_idx+1])
    fcs_reflection_response = clean_response_string(fcs_reflection_response, problem["answer"])
    fcs_reflection_response += "\n" + subsolutions[-1]

    problem["responses"]["fcs_reflection_trim"] = fcs_reflection_response

    # # Trimmed Only
    # trim_response = "\n".join(subsolutions[:-1])
    # trim_response = clean_response_string(trim_response, problem["answer"])
    # problem["responses"]["trim"] = trim_response

  for k in keys_to_filter:
    del dataset[k]

  with open(outfile, 'w', encoding='utf-8') as new_file:
      json.dump(dataset, new_file, ensure_ascii=False, indent=2)
  return dataset

def compute_token_usages(dataset, llm, outfile):
  tokenizer = llm.get_tokenizer()
  for key in tqdm(dataset, desc="Computing token usages", total=len(dataset)):
    problem = dataset[key]
    prompt_tokens = problem["token_usages"]["shortest"]["prompt_tokens"]
    problem["token_usages"]["fcs_trim"] = {
      "prompt_tokens": prompt_tokens,
      "completion_tokens": len(tokenizer(problem["responses"]["fcs_trim"]).input_ids)
    }
    problem["token_usages"]["fcs_reflection_trim"] = {
      "prompt_tokens": prompt_tokens,
      "completion_tokens": len(tokenizer(problem["responses"]["fcs_reflection_trim"]).input_ids)
    }
    # problem["token_usages"]["trim"] = {
    #   "prompt_tokens": prompt_tokens,
    #   "completion_tokens": len(tokenizer(problem["responses"]["trim"]).input_ids)
    # }

  with open(outfile, 'w', encoding='utf-8') as new_file:
      json.dump(dataset, new_file, ensure_ascii=False, indent=2)
  return dataset

def format_to_simpo(final_dataset, format, outfile):
  simpo_format = []
  for prompt in final_dataset:
    problem = final_dataset[prompt]
    simpo = {}
    simpo["conversations"] = [
      {
        "from": "system",
        "value": SYSTEM_PROMPT["NovaSky-AI/Sky-T1-32B-Preview"]
      },
      {
        "from": "human",
        "value": "Return your final response within \\boxed{{}}" + prompt
      }
    ]
    simpo["chosen"] = {
      "from": "gpt",
      "value": problem["responses"][format],
    }
    simpo["rejected"] = {
      "from": "gpt",
      "value": problem["responses"]["longest"]["content"]
    }
    simpo_format.append(simpo)

  with open(outfile, 'w', encoding='utf-8') as new_file:
      json.dump(simpo_format, new_file, ensure_ascii=False, indent=2)

# Generate output with a separator sequence (e.g., #####)
SUBPROBLEM_SPLIT_PROMPT = """
  You are given a reasoning sequence that attempts to solve a math problem.
  This sequence contains multiple proposed solutions, then provides a the final solution. 
  Each proposed solution within the sequence follows a different line of thought, usually to double check the answer. 
  Your objective is to identify these separate lines of thought and add the separator string '#####' between the separate lines of thought.
  This is important: Your response should be the original unchanged reasoning sequence, except for '#####' injected into the sequence between distinct lines of thought.
  Do NOT summarize portions of the reasoning sequence with '...'.

  Please keep the sequence that starts with '<|begin_of_solution|>' and ends with '<|end_of_solution|>' as 
  one single sequence with no '#####' 'inside of the sequence. Add the separator '#####' immediately before '<|begin_of_solution|>.

  Importantly, only use '#####' if a line of thought presents an answer. 
  If the line of thought does not include an answer, it cannot be considered a separate line of thought, and should not be separated.

  For example, if the input is:
  <|begin_of_thought|>The answer to 2+3 is 5. But wait, let me double check this. 
  If I have two apples and I am given three more apples, I now have 5 apples, so 5 seems like the right answer. 
  Alternatively, 2+3 is the same as 3+2, which is also 5.<|end_of_thought|>
  <|begin_of_solution|>The answer is 5<|end_of_solution|>. 

  Your output should be:
  <|begin_of_thought|>The answer to 2+3 is 5. 
  #####
  But wait, let me double check this. 
  If I have two apples and I am given three more apples, I now have 5 apples, so 5 seems like the right answer.
  ##### 
  Alternatively, 2+3 is the same as 3+2, which is also 5.<|end_of_thought|>
  #####
  <|begin_of_solution|>The answer is 5<|end_of_solution|>. 
"""

ANSWER_EXTRACTION_PROMPT = """
  You are given text of an attemp to solve a math problem. The text contains a final proposed answer to the math problem.

  The text also contains a string '#####' and after this string the ground truth answer is presented.

  Your objective is to determine whether the final proposed answer is equivalent to the ground truth answer.
  The proposed answer and ground truth answer may be in slightly different formats. For example, the proposed answer may be '1/2' but the ground truth is '0.5'.
  Equivalent answers in different formats should be treated as equivalent.
  If the text contains multiple proposed answers, use the final proposed answer.

  You should return only "True" if the proposed answer is equivalent to the ground truth answer and "False" if there is no proposed answer or if the proposed answer is not equivalent to the ground truth.
  Do NOT respond with anything at all except "True" or "False". 
  
  For example, if you are given:
  I believe 2+3 equals 5.
  #####
  The ground truth answer is five.

  Your response should be:
  True

  Another example, if you are given:
  I believe 2+2 equals 4. But wait, it is actually 5.
  #####
  The ground truth answer is five.

  Your response should be:
  True
"""

def main():
  parser = argparse.ArgumentParser(description="Tool to rewrite generated solutions for high-quality data generation.")
  parser.add_argument("--model", type=str, required=True, default="meta-llama/Llama-3.3-70B-Instruct", help="The model to run.")
  parser.add_argument("--tp", type=int, default=8, help="Tensor Parallelism Degree")
  parser.add_argument("--max_tokens", type=int, default=32768, help="Max tokens for the model.")
  args = parser.parse_args()

  filepath_root = "data/math5k/sky-t1-math5k"

  # Initialize model
  llm = LLM(model=args.model, tensor_parallel_size=args.tp)
  sampling_params = SamplingParams(max_tokens=args.max_tokens)

  # Load dataset to process
  dataset_filepath = filepath_root + "-original-answers.json"
  dataset = load_dataset(dataset_filepath)

  # Score full response
  scored_filepath = filepath_root + "-scored-solutions.json"
  scored_dataset = simple_score_solutions(dataset, scored_filepath)

  # Filter for shortest and longest correct solutions
  filtered_filepath = filepath_root + "-filtered-solutions.json"
  filtered_dataset = filter_solutions(scored_dataset, filtered_filepath)

  # Split short solution into subsolutions
  conversations = make_splitting_conversations(filtered_dataset, SUBPROBLEM_SPLIT_PROMPT)
  responses = llm.chat(messages=conversations, sampling_params=sampling_params, use_tqdm=True)
  subsolutions = split_solutions(responses)
  subsolution_filepath = filepath_root + "-subsolutions.json"
  split_dataset = add_split_to_dataset(filtered_dataset, subsolutions, subsolution_filepath)

  # Score subsolutions
  subscoring_conversations = make_subscoring_conversations(split_dataset, ANSWER_EXTRACTION_PROMPT)
  responses = llm.chat(messages=subscoring_conversations, sampling_params=sampling_params, use_tqdm=True)
  scored_subsolution_filepath = filepath_root + "-scored-subsolutions.json"
  scored_dataset = score_subsolutions(split_dataset, responses, scored_subsolution_filepath)

  # Create solution variants: FCS, FCS+Reflection, Trim-only
  variants_filepath = filepath_root + "-solution-variants.json"
  variants_dataset = build_solution_variants(scored_dataset, variants_filepath)

  final_filepath = filepath_root + "-final.json"
  final_dataset = compute_token_usages(variants_dataset, llm, final_filepath)

  simpo_filepath = filepath_root + "-simpo-fcs_trim.json"
  format_to_simpo(final_dataset, "fcs_trim", simpo_filepath)
  simpo_filepath = filepath_root + "-simpo-fcs_reflection_trim.json"
  format_to_simpo(final_dataset, "fcs_reflection_trim", simpo_filepath)
  # simpo_filepath = filepath_root + "-simpo-trim.json"
  # format_to_simpo(final_dataset, "trim", simpo_filepath)

  # simpo_dataset = load_dataset(filepath_root + "-simpo-fcs_reflection_trim.json")
  # print(len(simpo_dataset))


if __name__ == "__main__":
    main()
