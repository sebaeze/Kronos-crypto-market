class Config:
    def __init__(self):
        # --- Core Training Variables ---
        self.seed = 42
        self.accumulation_steps = 1   
        self.log_steps = 10           
        self.log_interval = 10        
        self.n_train_iter = 1000      
        self.n_valid_iter = 200       
        
        # --- Paths and Instrument ---
        # Relative to the Kronos-crypto-market folder in Workbench
        self.dataset_path = "./finetune_csv/data/kronos_combined_1m_training_data.csv"
        self.instrument = 'GUAUSDT'

        self.dataset_begin_time = "2020-01-01"
        self.dataset_end_time = '2030-01-01'

        # --- Model Architecture ---
        self.lookback_window = 400   
        self.predict_window = 60    
        self.max_context = 512       

        self.feature_list = ['open', 'high', 'low', 'close', 'volume']
        self.time_feature_list = ['minute', 'hour', 'weekday', 'day', 'month']

        # --- Hardware & Optimizers ---
        self.device = 'cuda'
        self.epochs = 3
        self.batch_size = 8
        self.tokenizer_learning_rate = 1e-5   
        self.predictor_learning_rate = 1e-5   
        self.adam_weight_decay = 1e-4
        self.adam_beta1 = 0.9           
        self.adam_beta2 = 0.999         
        
        # --- Save Paths ---
        self.save_path = "./output_models"
        self.tokenizer_save_folder_name = "tokenizer_finetuned"
        self.predictor_save_folder_name = "predictor_finetuned"

        self.pretrained_tokenizer_path = "NeoQuasar/Kronos-Tokenizer-base"
        self.pretrained_predictor_path = "NeoQuasar/Kronos-base"

        self.finetuned_tokenizer_path = f"{self.save_path}/{self.tokenizer_save_folder_name}/checkpoints/best_model"
        self.finetuned_predictor_path = f"{self.save_path}/{self.predictor_save_folder_name}/checkpoints/best_model"

        # --- Extras ---
        self.use_comet = False      
        self.backtest_benchmark = "GUAUSDT"