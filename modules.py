import os
import time
from configparser import ConfigParser

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

config = ConfigParser()
config.read("config.ini")

OPENAI_SECRET_KEY = config["openai"]["secret_key"]
MODEL = config["openai"]["model"]
TEMPERATURE = float(config["openai"]["temperature"])

os.environ["OPENAI_API_KEY"] = OPENAI_SECRET_KEY


class ReverseChainBaseClass:
    def __init__(self, model: str, temperature: float) -> None:
        self.model = model
        self.temperature = temperature
        self.llm = OpenAI(
            model_name=self.model,
            temperature=self.temperature,
        )
        self.template = ""

    def get_context_from_retriver(self, query: str, db):
        documents = db.retrieve_using_similarity_search(query, top_k=10)
        if documents is not None:
            _document = []
            for document in documents:
                _document.append(
                    f"\nNext API:\n{document.page_content}\nSource: {document.metadata['source']}\n"
                )
            context = " ".join(_document)
            return context
        return "No similar API Found!"

    def get_prompt(self, query: str, context: str) -> str:
        prompt = PromptTemplate(
            input_variables=["query", "context"], template=self.template
        )
        return prompt.format(query=query, context=context)


class FinalAPISelector(ReverseChainBaseClass):
    def __init__(self, model: str, temperature: float) -> None:
        super(FinalAPISelector, self).__init__(model, temperature)
        self.template = """ 
        We have below APIs that are similar to query:
        =====
        {context}
        =====
        1. Search for words like summarize, prioritize, my id, current sprint and select the api according to that.
        2. Try to divide the query in smaller tasks just like a human would do, now which final API should we use for this instruction? if no api can be used just return None in answer
        3. Only return API name in answer, donot return anything else.
        4. Only return one api, that is final api that will be used in the answer
        5. return the answer as a json object where key is api_name and key is the api name and a key data_source and value as the source of the file.
        Never give argument_name as the api name.
        If someone is saying: "{query}"
        Output:
        """

    def select_api_from_query(self, query: str, db) -> str:
        time.sleep(2)
        context = self.get_context_from_retriver(query, db)
        prompt = self.get_prompt(query, context=context)
        response = self.llm(prompt)
        return response


class ArgumentExtractor(ReverseChainBaseClass):
    def __init__(self, model: str, temperature: float) -> None:
        super(ArgumentExtractor, self).__init__(model, temperature)
        # Use it to get any named arguments, that may be required in the current API call.
        self.template = """
        Available Arguments:
        {api_response_variables}

        You are an argument extractor. For each argument, you need to
        determine whether you can extract the value from user input
        directly or from available arguments above or you need to use an API to get the value. 
        The output should be in Json format, key is the argument, and value is the
        value of argument. Importantly, return "" if you cannot get
        value.
        The api documentation is as below, use the context of the API to extract
        arguments that can be extracted from the user input or available arguments above and feeded in the API.
        For the arguments that have required = false in the api documentation, try to find the value in the user input
        if not found return \"RequiredFalse\" in front of them in the output in string format in the json object
        strictly if API in below API documentation doesnot use any arguments then, just return empty json object.

        API Documentation: 
        {context}
        ......
        Now, Let's start.

        If someone is saying: "{query}"

        Arguments :
        """

    def get_prompt(self, query: str, context: str, api_response_variables) -> str:
        # print(api_response_variables)
        prompt = PromptTemplate(
            input_variables=["query", "context", "api_response_variables"],
            template=self.template,
        )
        return prompt.format(
            query=query, context=context, api_response_variables=api_response_variables
        )

    def get_arguments_from_query(
        self, query: str, db, api_documentation, api_response_variables
    ):
        time.sleep(2)
        prompt = self.get_prompt(
            query=query,
            context=api_documentation,
            api_response_variables=api_response_variables,
        )
        response = self.llm(prompt)
        return response


class SubAPISelector(ReverseChainBaseClass):
    def __init__(self, model: str, temperature: float) -> None:
        super().__init__(model, temperature)
        self.template = """
        Required argument: {required_argument} 
        context: {context}
        Given the context above, give the API name that can give the reqiured_argument as output of the API.
        if no api can be used just return None in answer
        Only return API name in answer, donot return anything else.
        return the answer as a json object where key is api_name and key is the api name and a key data_source and value as the source of the file.
        """

    def get_api_from_argument(self, db, required_argument: str) -> str:
        context = self.get_context_from_retriver(required_argument, db)
        prompt = self.get_prompt(context=context, required_argument=required_argument)
        response = self.llm(prompt)
        return response

    def get_context_from_retriver(self, query: str, db):
        return db.retrieve_using_similarity_search(query, top_k=5)

    def get_prompt(self, context: str, required_argument: str) -> str:
        time.sleep(2)
        prompt = PromptTemplate(
            input_variables=["context", "required_argument"], template=self.template
        )
        return prompt.format(context=context, required_argument=required_argument)
