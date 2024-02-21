from langchain.output_parsers import StructuredOutputParser, ResponseSchema

agent_output_response_schema = [
    ResponseSchema(
        name="next_action",
        type="string",
        description="Indicates the next tool or agent to call. If there are none or if it is the end return 'END', else return agent or tool name."
    ),
    ResponseSchema(
        name="action_type",
        type="string",
        description="Indicates the type of action to be performed. It can be 'tool', 'agent' or None."
    ),
    ResponseSchema(
        name="action_parameters",
        type="object",
        description="The parameters for the next action. It can be empty."
    ),
]

agent_output_parser = StructuredOutputParser.from_response_schemas(agent_output_response_schema)

agent_output_instructions = agent_output_parser.get_format_instructions()