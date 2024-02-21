from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_openai import ChatOpenAI
from catalog_ai.output_parsers.agents import agent_output_parser, agent_output_instructions, agent_output_response_schema

from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from typing import List

text = """
You are an expert in Hugging Face Hub datasets. 
Your primary function is to understand the user's input and generate an accurate search string
 for the `list_datasets` method in the Hugging Face Hub SDK. The search parameter is critical,
   as it affects the entire flow by returning datasets that mention the text.
    
You excel at interpreting user input to formulate relevant search queries,
  even when the user does not explicitly provide a search string. 

Return to the supervisor after generating the search string.
"""

custom_response_schema = [
    ResponseSchema(
        name="search_string",
        type="string",
        description="The search string to be used in the `list_datasets` method in the Hugging Face Hub SDK."
    )
]

for schema in agent_output_response_schema:
    custom_response_schema.append(schema)


custom_response_parser = StructuredOutputParser.from_response_schemas(
    response_schemas= custom_response_schema
)

custom_output_instructions = custom_response_parser.get_format_instructions()

prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            text
        ),
        SystemMessagePromptTemplate.from_template(
            "Agents available for this task: {agents}"    
        ),
        SystemMessagePromptTemplate.from_template(
            "user_question: {user_question}"
        ),
        SystemMessagePromptTemplate.from_template(
            "Output instructions: {output_instructions}"
        )
    ],
    input_variables=['user_question'],
    partial_variables={'output_instructions':custom_output_instructions,
                       'agents': ['supervisor']
                       }
)

llm = ChatOpenAI(model="gpt-3.5-turbo")

search_string_extractor_runnable = prompt | llm | custom_response_parser