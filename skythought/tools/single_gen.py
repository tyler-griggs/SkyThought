from vllm import LLM, SamplingParams


llm = LLM(model="NovaSky-AI/Sky-T1-32B-Preview", tensor_parallel_size=8)
sampling_params = SamplingParams(max_tokens=1000, temperature=0)

SYSTEM_PROMPT = "Your role as an assistant involves thoroughly exploring questions through a systematic long \
        thinking process before providing the final precise and accurate solutions. This requires \
        engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, \
        backtracing, and iteration to develop well-considered thinking process. \
        Please structure your response into two main sections: Thought and Solution. \
        In the Thought section, detail your reasoning process using the specified format: \
        <|begin_of_thought|> {thought with steps separated with '\n\n'} \
        <|end_of_thought|> \
        Each step should include detailed considerations such as analisying questions, summarizing \
        relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining \
        any errors, and revisiting previous steps. \
        In the Solution section, based on various attempts, explorations, and reflections from the Thought \
        section, systematically present the final solution that you deem correct. The solution should \
        remain a logical, accurate, concise expression style and detail necessary step needed to reach the \
        conclusion, formatted as follows: \
        <|begin_of_solution|> \
        {final formatted, precise, and clear solution} \
        <|end_of_solution|> \
        Now, try to solve the following question through the above guidelines:"

conversations = []
conversations.append([
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "Are sharks older than the moon?"}
])
responses = llm.chat(messages=conversations, sampling_params=sampling_params, use_tqdm=True)
print(responses)