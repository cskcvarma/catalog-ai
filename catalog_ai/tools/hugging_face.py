import os
from huggingface_hub import HfApi
from langchain.tools import tool

HF_TOKEN = os.environ["HF_TOKEN"]
HF_ENDPOINT = os.environ["HF_ENDPOINT"]

hf_api = HfApi(
    endpoint=HF_ENDPOINT,
    token=HF_TOKEN
)

# @tool("get_datasets", return_direct=True)
def get_datasets(search_string: str) -> list:
    """
    Use the Hugging Face Hub SDK to retrieve a list of datasets matching the search string.
    """
    datasets = hf_api.list_datasets(search=search_string)
    return list(datasets)