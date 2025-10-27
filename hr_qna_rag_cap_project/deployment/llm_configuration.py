
#Libraries for downloading and loading the llm
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

class LLMConfiguration:

    def __init__(self):
        pass

    # Define filename to download models from Hugging Face model hub.
    def prepareLlmInstanceAndGetInstance(self):
        model_name_or_path = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
        model_basename = "mistral-7b-instruct-v0.2.Q6_K.gguf"

        # Using hf_hub_download to download a model from the Hugging Face model hub
        # The repo_id parameter specifies the model name or path in the Hugging Face repository
        # The filename parameter specifies the name of the file to download
        model_path = hf_hub_download(
            repo_id=model_name_or_path,
            filename=model_basename
        )

        #Initializing LLM model basic configurations
        lcpp_llm = Llama(
            model_path=model_path,
            n_threads=2,      # CPU cores
            n_batch=512,      # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
            n_gpu_layers=30,  # Change this value based on your model and your GPU VRAM pool.
            n_ctx=4096,       # Context window
        )

        return lcpp_llm


