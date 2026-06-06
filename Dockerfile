FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Install required system tools and Google Cloud CLI
RUN apt-get update && apt-get install -y git zip curl gnupg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add - && \
    apt-get update -y && apt-get install google-cloud-cli -y

# Set the working directory
WORKDIR /workspace

# Clone the custom training branch
RUN git clone -b cgp_cloud_opnusdt https://github.com/sebaeze/Kronos-crypto-market .

# Install Python requirements
RUN pip install --no-cache-dir -r requirements.txt comet_ml pandas google-cloud-storage

# Copy and configure the entrypoint script
COPY entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

# Ensure Kronos architecture resolves correctly in the Python path
ENV PYTHONPATH="/workspace"

ENTRYPOINT ["/workspace/entrypoint.sh"]