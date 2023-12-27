from modules import FinalAPISelector, ArgumentExtractor, SubAPISelector
from typing import Dict, Any
from executor import Executor
from result_formatter import ResultFormatter
from retriever import VectorDataBase
from planner import Planner
from configparser import ConfigParser
import os
import re
import json
import time
from collections import deque
import logging

import warnings

warnings.filterwarnings("ignore")

config = ConfigParser()
config.read("config.ini")

DATA_PATH = config["faiss"]["data"]

OPENAI_SECRET_KEY = config["openai"]["secret_key"]
MODEL = config["openai"]["model"]
TEMPERATURE = float(config["openai"]["temperature"])

MAX_EXECUTION_TIME = config["reverse_gpt"]["max_execution_time"]
MAX_ITERATIONS = config["reverse_gpt"]["max_iterations"]

QUERY = config["query"]["query"]

os.environ["OPENAI_API_KEY"] = OPENAI_SECRET_KEY


def simpleFormatter(context, prev_api_mapping):
    # print(prev_api_mapping)
    response = []
    for api in context:
        res = {}
        res["tool_name"] = api["api_name"]
        res["arguments"] = []

        for key, value in api["arguments"].items():
            for _, v in prev_api_mapping.items():
                if value == v[1]:
                    res["arguments"].append(
                        {"argument_name": key, "argument_value": v[0]}
                    )
            else:
                res["arguments"].append({"argument_name": key, "argument_value": value})
        response.append(res)
    return response


def _should_end(result):
    if re.search("Final Answer", result):
        return True
    return False


if __name__ == "__main__":
    vector_db = VectorDataBase()
    vector_db.load_db()

    logging.basicConfig(
        level=logging.INFO,
        filename="logs/run.log",
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger()

    query = QUERY

    time_elapsed = 0.0
    start_time = time.time()

    api_selector = FinalAPISelector(MODEL, TEMPERATURE)
    argument_extractor = ArgumentExtractor(MODEL, TEMPERATURE)
    sub_api_selector = SubAPISelector(MODEL, TEMPERATURE)
    planner = Planner(MODEL, TEMPERATURE)
    executor = Executor()
    formatter = ResultFormatter(MODEL, TEMPERATURE)

    planner_history = []

    plan = planner.run(inputs={"input": query, "history": planner_history})

    logger.info(f'Planner Output: {plan["result"]}')

    if plan["result"] == "[]":
        print("[]")
        exit()

    prev_table = {}
    api_tree = []
    prev_api_mapping = {}
    while not _should_end(plan["result"]):
        ## getting the root api
        api = api_selector.select_api_from_query(query=plan["result"], db=vector_db)
        api = json.loads(api)

        logger.info(f"API Selector: {api}")

        if api == "None":
            print("[]")
            exit()

        with open(api["data_source"], "r") as f:
            api_documentation = f.read()

        arguments = argument_extractor.get_arguments_from_query(
            query=query,
            db=vector_db,
            api_documentation=api_documentation,
            api_response_variables=prev_table,
        )

        # print(f"Arguments: {arguments}")

        if len(arguments) < 5:
            arguments = {}
        else:
            arguments = json.loads(arguments)

        keys = list(arguments.keys())
        for k in keys:
            if arguments[k] == "RequiredFalse":
                arguments.pop(k)

        stack = deque()

        logger.info(f"Argument Selector: {arguments}")

        for key, value in arguments.items():
            if value is None:
                stack.append(key)

        while stack:
            next_required_argument = stack.pop()
            api = sub_api_selector.get_api_from_argument(
                required_argument=next_required_argument, db=vector_db
            )
            api = json.loads(api)

            if api == "None":
                print("[]")
                exit()

            logger.info(f"API Selector: {api}")

            with open(api["data_source"], "r") as f:
                api_documentation = f.read()

            arguments = argument_extractor.get_arguments_from_query(
                query=query,
                db=vector_db,
                api_documentation=api_documentation,
                api_response_variables=prev_table,
            )

            arguments = json.loads(arguments)

            keys = list(arguments.keys())
            for k in keys:
                if arguments[k] == "RequiredFalse":
                    arguments.pop(k)

            for key, value in arguments.items():
                if value is None:
                    stack.append(key)

            logger.info(f"Argument Selector: {arguments}")

        function_json = {"api_name": api["api_name"], "arguments": arguments}

        logger.info(f"JSON: {function_json}")

        response = executor.run(function_json)
        status_code = response.pop("status")
        if status_code != 200:
            execution_response_msg = (
                "Unsuccessful attempt, cause: " + response.get("error", "unknown") + "!"
            )
        else:
            execution_response_msg = response.pop("message")

        api_call_summary = {
            "sequence_no": len(api_tree),
            "api_name": api["api_name"],
            "arguments": arguments,
            "output": response,
        }
        api_tree.append(api_call_summary)

        for k, v in response.items():
            prev_table[k] = v
            prev_api_mapping[k] = (f"$$PREV[{len(api_tree) - 1}]", v)

        planner_history.append((plan["result"], execution_response_msg))
        plan = planner.run(inputs={"input": query, "history": planner_history})

        logger.info(f"Planner: {plan}")

    time_elapsed = time.time() - start_time

    #formatted_result = formatter.run(api_tree, prev_table) # code to format using llm
    formatted_result = simpleFormatter(
        api_tree, prev_api_mapping
    )  # Simple formatter does basic mapping

    with open("output/output.json", "w") as f:
        f.write(json.dumps(formatted_result, indent=4))
        f.close()

    logger.info(f"TIME: {time_elapsed}")
