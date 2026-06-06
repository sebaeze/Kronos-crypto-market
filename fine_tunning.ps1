# --- Configuration Variables ---
$PROJECT_ID = "YOUR_GCP_PROJECT_ID" 
$REGION = "us-central1"
$REPO_NAME = "kronos-repo"
$IMAGE_NAME = "kronos-trainer:latest"
$IMAGE_URI = "$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/$IMAGE_NAME"
$BUCKET_NAME = "YOUR_GCS_BUCKET_NAME"

# --- Step 1: Create Artifact Registry (Run once per project) ---
gcloud artifacts repositories create $REPO_NAME `
    --repository-format=docker `
    --location=$REGION `
    --description="Docker repository for Kronos ML jobs"

# --- Step 2: Configure Docker Auth ---
gcloud auth configure-docker $REGION-docker.pkg.dev --quiet

# --- Step 3: Build and Push the Docker Image ---
docker build -t $IMAGE_URI .
docker push $IMAGE_URI

# --- Step 4: Trigger the Vertex AI Custom Training Job ---
gcloud ai custom-jobs create `
    --region=$REGION `
    --display-name="kronos-opnusdt-finetune" `
    --worker-pool-spec="machine-type=g2-standard-4,accelerator-type=NVIDIA_L4,accelerator-count=1,replica-count=1,container-image-uri=$IMAGE_URI,env=BUCKET_NAME=$BUCKET_NAME" `
    --project=$PROJECT_ID