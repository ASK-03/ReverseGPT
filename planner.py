from typing import Any, Dict, List, Optional, Tuple
import time
import re
from configparser import ConfigParser
import os

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain.llms.base import BaseLLM
from langchain.llms import OpenAI


config = ConfigParser()
config.read("config.ini")

OPENAI_SECRET_KEY = config["openai"]["secret_key"]
os.environ["OPENAI_API_KEY"] = OPENAI_SECRET_KEY

icl_examples = {
    "devrev": """Example 1:
User query: Prioritize my P0 issues and add them to the current sprint.
Plan step 1: Get the user by user id.
API response: Successfully identified the user.
Plan step 2: Retrieve a list of P0 issues owned by the user.
API response: Successfully obtained a list of P0 issues owned by the user.
Plan step 3: Prioritize the retrieved P0 issues.
API response: Successfully prioritized the list of P0 issues.
Plan step 4: Get the current sprint ID.
API response: Successfully obtained the current sprint ID.
Plan step 5: Add the prioritized P0 issues to the current sprint.
API response: Successfully added the prioritized P0 issues to the current sprint.
Thought: I am finished executing a plan and have prioritized P0 issues and added them to the current sprint.
Final Answer: P0 issues have been successfully prioritized and added to the current sprint.

Example 2:
User query: Summarize issues similar to don:core:dvrv-us-1:devo/0:issue/1.
Plan step 1: get the work items similar to don:core:dvrv-us-1:devo/0:issue/1.
API response: Successfully called the tool responsible for finding similar work items with the necessary arguments.
Plan step 2: Summarize the retrieved work items using the summarization tool.
API response: Successfully called the tool responsible for summarizing objects with the required arguments, utilizing the information from the previous step.
Thought: I am finished executing a plan and have summarized issues similar to don:core:dvrv-us-1:devo/0:issue/1.
Final Answer: Here is the summary of issues similar to don:core:dvrv-us-1:devo/0:issue/1.

Example 3:
User query: Summarize high severity tickets from the customer UltimateCustomer.
Plan step 1: Search for the customer with the name "UltimateCustomer."
API response: Successfully searched for the customer with the name "UltimateCustomer."
Plan step 2: Retrieve a list of high severity tickets associated with the customer "UltimateCustomer".
API response: Successfully obtained a list of high severity tickets associated with the customer.
Plan step 3: Summarize the retrieved high severity tickets.
API response: Successfully summarized the list of high severity tickets.
Thought: I am finished executing a plan and have summarized high severity tickets from the customer "UltimateCustomer."
Final Answer: High severity tickets from the customer "UltimateCustomer" have been successfully summarized.
""",
}


PLANNER_PROMPT = """
You are an agent that plans solution to user queries.
You should always give your plan in natural language.
Another model will receive your plan and find the right API calls and give you the result in natural language.
If you assess that the current plan has not been fulfilled, you can output "Continue" to let the API selector select another API to fulfill the plan.
If you think you have got the final answer or the user query has been fulfilled, just output the answer immediately. If the query has not been fulfilled, you should continue to output your plan.
In most case, search, filter, and sort should be completed in a single step.
The plan should be as specific as possible. It is better not to use pronouns in plan, but to use the corresponding results obtained previously. For example, instead of "Get the most popular movie directed by this person", you should output "Get the most popular movie directed by Martin Scorsese (1032)". If you want to iteratively query something about items in a list, then the list and the elements in the list should also appear in your plan.
The plan should be straightforward. If you want to search, sort or filter, you can put the condition in your plan. For example, if the query is "Who is the lead actor of In the Mood for Love (id 843)", instead of "get the list of actors of In the Mood for Love", you should output "get the lead actor of In the Mood for Love (843)".
Divide the task in subtask as how a human would do it in sequential manner.
Donot miss subtask in the plan.

Starting below, you should follow this format:

User query: the query a User wants help with related to the API.
Plan step 1: the first step of your plan for how to solve the query
API response: the result of executing the first step of your plan, including the specific API call made.
Plan step 2: based on the API response, the second step of your plan for how to solve the query. If the last step result is not what you want, you can output "Continue" to let the API selector select another API to fulfill the plan. For example, the last plan is "add a song (id xxx) in my playlist", but the last step API response is calling "GET /me/playlists" and getting the id of my playlist, then you should output "Continue" to let the API selector select another API to add the song to my playlist. Pay attention to the specific API called in the last step API response. If a inproper API is called, then the response may be wrong and you should give a new plan.
API response: the result of executing the second step of your plan
... (this Plan step n and API response can repeat N times)
Thought: I am finished executing a plan and have the information the user asked for or the data the used asked to create
Final Answer: the final output from executing the plan

Use examples to create and improve the plan.
Use the same language as used in the examples.
The api give the result in directly useable format.
Before giving final answer, check if the user query has been fulfilled. if not then
Give the steps in continuations. Fill the next steps that can be taken to fulfil the input query.
If it the query is not related to technology, return [] as the result.

Examples:
{icl_examples}

Begin!

User query: {input}
Plan step 1: {agent_scratchpad}"""


class Planner:
    llm: BaseLLM
    planner_prompt: str
    output_key: str = "result"

    def __init__(self, model, temperature, planner_prompt=PLANNER_PROMPT) -> None:
        self.model = model
        self.temperature = temperature
        self.llm = OpenAI(
            model_name=self.model,
            temperature=self.temperature,
        )
        self.planner_prompt = PLANNER_PROMPT

    @property
    def _chain_type(self) -> str:
        return "ReverseGPT Planner"

    @property
    def input_keys(self) -> List[str]:
        return ["input"]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    @property
    def observation_prefix(self) -> str:
        """Prefix to append the observation with."""
        return "API response: "

    @property
    def llm_prefix(self) -> str:
        """Prefix to append the llm call with."""
        return "Plan step {}: "

    @property
    def _stop(self) -> List[str]:
        return [
            f"\n{self.observation_prefix.rstrip()}",
            f"\n\t{self.observation_prefix.rstrip()}",
        ]

    def _construct_scratchpad(self, history: List[Tuple[str, str]]) -> str:
        if len(history) == 0:
            return ""
        scratchpad = ""
        for i, (plan, execution_res) in enumerate(history):
            scratchpad += self.llm_prefix.format(i + 1) + plan + "\n"
            scratchpad += self.observation_prefix + execution_res + "\n"
        return scratchpad

    def run(self, inputs: Dict[str, List[Tuple[str, str]]]) -> Dict[str, str]:
        time.sleep(2)
        scratchpad = self._construct_scratchpad(inputs["history"])
        # print("Scrachpad: \n", scratchpad)
        planner_prompt = PromptTemplate(
            template=self.planner_prompt,
            partial_variables={
                "agent_scratchpad": scratchpad,
                "icl_examples": icl_examples["devrev"],
            },
            input_variables=["input"],
        )
        planner_chain = LLMChain(llm=self.llm, prompt=planner_prompt)
        planner_chain_output = planner_chain.run(input=inputs["input"], stop=self._stop)

        planner_chain_output = re.sub(
            r"Plan step \d+: ", "", planner_chain_output
        ).strip()

        return {"result": planner_chain_output}


if __name__ == "__main__":
    planner = Planner("gpt-3.5-turbo", temperature=0.1)
    query = "List all high severity tickets coming in from slack from customer Cust123 and generate a summary of them"
    planner_history = []
    plan = planner.run(inputs={"input": query, "history": planner_history})
    print(plan)
