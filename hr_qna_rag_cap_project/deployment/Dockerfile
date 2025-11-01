# Use a nvidia cuda supported ubuntu base image
FROM nvidia/cuda:12.9.0-runtime-ubuntu22.04

# install python ninja git and nvidia toolkit
RUN apt-get update && apt-get install -y python3 python3-pip ninja-build git nvidia-container-toolkit

# Set the working directory inside the container to /app
WORKDIR /app

# Copy all files from the current directory on the host to the container's /app directory
COPY . .

# Adding the cuda and cmake flags to rebuld llama cpp with cublas on
ENV CMAKE_ARGS="-DLLAMA_CUBLAS=on"
ENV FORCE_CMAKE=1

RUN pip3 install llama-cpp-python==0.1.85 --force-reinstall --no-cache-dir -q

# Install Python dependencies listed in requirements.txt
RUN pip3 install -r requirements.txt

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user . $HOME/app

# Define the command to run the Streamlit app on port "8501" and make it accessible externally
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
