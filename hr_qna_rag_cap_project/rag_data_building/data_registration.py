from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os


repo_id = "rishabhsinghjk/HR-QnA-rag-dataspace"
repo_type = "dataset"

# Initialize HF API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the space exists if not then create it
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"dataSpace '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"dataSpace '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"dataSpace '{repo_id}' created.")

# Step 2: upload the content to hf dataset space
api.upload_folder(
    folder_path="hr_qna_rag_cap_project/data_registration",
    repo_id=repo_id,
    repo_type=repo_type
)
