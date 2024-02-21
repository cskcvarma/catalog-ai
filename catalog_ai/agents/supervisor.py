from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from catalog_ai.output_parsers.agents import agent_output_parser, agent_output_instructions


text = """
As the Dataset Supervisor,
 I assist users in identifying relevant datasets for their needs. 
 When a user presents a query, I first collaborate with the "search_string_extractor" agent. This is important, I need to generate the search string first
 to develop an effective search string.
 
 Once I have the search string, 
 I proceed with the "get_datasets" tool to acquire a list of datasets matching the user's criteria. 
 
 Should the initial list exceed 10 datasets,
   I utilize the "ask_user" function to involve the user in refining their search.
     This can be achieved through filtering options based on task categories, task IDs, or dataset size.
   Upon recieving the user's input, I use the "filter_data" tool to ensure the list becomes more focused and manageable.
   This iterative process of refinement continues until the dataset list is concise (fewer than 10 items) or until the user decides against further narrowing. I handle inputs like "user_question," "chat_history," "length_of_data," and "search_string." The absence of any of these inputs signals that the respective step is pending.

 My goal is to guide users effectively through the dataset selection process, ensuring they receive a tailored list that aligns with their specific project requirements.
"""

text2 = """
As a supervisor who has no knowledge of the user's question,
 I need to use the agents at my disposal to generate a search string.

 Upon collecing the search string, I will use the "get_datasets" tool to acquire a list of datasets matching the user's criteria.

 and they return with "Data collection is completed"

"""

prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            text2
        ),
        SystemMessagePromptTemplate.from_template(
            "Agents available for this task: {agents}"    
        ),
        SystemMessagePromptTemplate.from_template(
            "Tools available for this task: {tools}"
        ),
        HumanMessagePromptTemplate.from_template(
            "user question: {user_question}"
        ),
        SystemMessagePromptTemplate.from_template(
            """
            Available data:
            - search_string: {search_string}
            - length_of_data: {length_of_data}
            """
        ),
        SystemMessagePromptTemplate.from_template(
            "Output instructions: {output_instructions}"
        )
    ],
    input_variables=['user_question', 'search_string', 'length_of_data'],
    partial_variables={'output_instructions':agent_output_instructions,
                        'agents': ['search_string_extractor'],
                          'tools': ['get_datasets']}
)

llm = ChatOpenAI(model="gpt-3.5-turbo")

supervison_runnable = prompt | llm | agent_output_parser