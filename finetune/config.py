import os

class Config:
    """
    Configuration class for the entire project.
    Optimized for GCP Vertex AI and 1-Minute Crypto Data.
    """

    def __init__(self):
        # =================================================================
        # Data & Feature Parameters
        # =================================================================
        # Set to the GCS mounted directory where your CSV file is located.
        # Note: If you are bypassing Qlib, ensure your dataset loader expects this CSV.
        self.dataset_path = "/gcs/kronos-1m-workspace/dataset/kronos_combined_1m_training_data.csv"
        self.instrument = 'GUAUSDT'

        # Set a broad time range that encompasses your 2026 data
        self.dataset_begin_time = "2026-03-16"
        self.dataset_end_time = '2026-05-30'

        # Sliding window parameters for creating samples.
        # For 1-minute scalping, looking back 400 minutes to predict the next 60 minutes.
        self.lookback_window = 400   
        self.predict_window = 60    
        self.max_context = 512       # Do not change this; Kronos architecture limit.

        # Aligned exactly with the CSV output from our conversion script
        self.feature_list = ['open', 'high', 'low', 'close', 'volume']
        
        # Time-based features. 
        self.time_feature_list = ['minute', 'hour', 'weekday', 'day', 'month']

        # =================================================================
        # Hardware & Training Hyperparameters
        # =================================================================
        self.device = 'cuda'        # Force GPU usage for Vertex AI
        self.epochs = 3             # Keep low (3-5) to avoid overfitting 1m noise
        self.batch_size = 8         # Safe batch size for an NVIDIA L4 24GB GPU
        self.learning_rate = 1e-5   # Conservative learning rate
        
        # =================================================================
        # Model & Checkpoint Paths
        # =================================================================
        # Cloud Storage output directory for saved model weights
        self.save_path = "/gcs/kronos-1m-workspace/output_models"
        
        # Subfolders for the two training stages
        self.tokenizer_save_folder_name = "tokenizer_finetuned"
        self.predictor_save_folder_name = "predictor_finetuned"

        # The base foundation models to pull directly from Hugging Face
        self.pretrained_tokenizer_path = "NeoQuasar/Kronos-Tokenizer-base"
        self.pretrained_predictor_path = "NeoQuasar/Kronos-base"

        # Paths to the fine-tuned models, derived automatically during training
        self.finetuned_tokenizer_path = f"{self.save_path}/{self.tokenizer_save_folder_name}/checkpoints/best_model"
        self.finetuned_predictor_path = f"{self.save_path}/{self.predictor_save_folder_name}/checkpoints/best_model"

        # =================================================================
        # Logging & Tracking
        # =================================================================
        self.use_comet = False      # Disabled to avoid requiring third-party API keys

        # =================================================================
        # Backtesting Parameters (Used only if running qlib_test.py)
        # =================================================================
        self.backtest_n_symbol_hold = 1 
        self.backtest_n_symbol_drop = 0  
        self.backtest_hold_thresh = 5  
        self.inference_T = 1.0          # Temperature for inference
        self.inference_top_p = 0.9
        self.inference_top_k = 0
        self.inference_sample_count = 5
        self.backtest_batch_size = 1000
        self.backtest_benchmark = "GUAUSDT"