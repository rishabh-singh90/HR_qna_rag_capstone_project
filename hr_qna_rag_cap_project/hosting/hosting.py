from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="hr_qna_rag_cap_project/deployment",     # the local folder containing your files
    repo_id="rishabhsinghjk/HR-QnA-rag",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
