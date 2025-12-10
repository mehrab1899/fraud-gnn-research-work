import os
import yaml
from pathlib import Path
from dotenv import load_dotenv


class Config:
    """Configuration loader for fraud detection GNN project."""
    
    _env_loaded = False
    _config = {}
    
    @classmethod
    def load_env(cls):
        """Load environment variables from .env file."""
        if not cls._env_loaded:
            env_path = Path(__file__).parent / ".env"
            if env_path.exists():
                load_dotenv(env_path)
                cls._env_loaded = True
            else:
                raise FileNotFoundError(f".env file not found at {env_path}")
    
    @classmethod
    def load_config_yaml(cls):
        """Load configuration from config.yaml file."""
        yaml_path = Path(__file__).parent / "config.yaml"
        if yaml_path.exists():
            with open(yaml_path, 'r') as f:
                cls._config = yaml.safe_load(f) or {}
        else:
            raise FileNotFoundError(f"config.yaml file not found at {yaml_path}")
    
    @classmethod
    def get_dataset_path(cls, dataset_key):
        """
        Get dataset path from environment variables.
        
        Args:
            dataset_key: One of 'financial', 'credit', 'loans', 'transaction'
        
        Returns:
            Path to the dataset file
        """
        cls.load_env()
        
        key_map = {
            'financial': 'FINANCIAL_FRAUD_DATASET_PATH',
            'credit': 'GERMAN_CREDIT_DATASET_PATH',
            'loans': 'CREDIT_RISK_DATASET_PATH',
            'transaction': 'TRANSACTION_DATASET_PATH'
        }
        
        env_key = key_map.get(dataset_key)
        if not env_key:
            raise ValueError(f"Unknown dataset key: {dataset_key}")
        
        path = os.getenv(env_key)
        if not path:
            raise ValueError(f"Environment variable {env_key} not set")
        
        return path
    
    @classmethod
    def get_hyperparams(cls):
        """Get training hyperparameters from environment variables."""
        cls.load_env()
        
        return {
            'hidden_dim': int(os.getenv('HIDDEN_DIM', '16')),
            'max_trx_per_company': int(os.getenv('MAX_TRX_PER_COMPANY', '8')),
            'limit_per_entity': int(os.getenv('LIMIT_PER_ENTITY', '8')),
            'make_cliques': os.getenv('MAKE_CLIQUES', 'False').lower() == 'true',
            'epochs': int(os.getenv('EPOCHS', '8')),
            'learning_rate': float(os.getenv('LEARNING_RATE', '0.001')),
            'weight_decay': float(os.getenv('WEIGHT_DECAY', '0.0001')),
            'quick_run': os.getenv('QUICK_RUN', 'True').lower() == 'true',
            'quick_run_subsample': float(os.getenv('QUICK_RUN_SUBSAMPLE', '0.05')),
            'quick_shap_samples': int(os.getenv('QUICK_SHAP_SAMPLES', '10')),
            'loss_type': os.getenv('LOSS_TYPE', 'focal'),
        }
    
    @classmethod
    def get_output_paths(cls):
        """Get output directory paths."""
        cls.load_env()
        
        output_dir = os.getenv('EXPLANATION_OUTPUT_DIR', './explanations')
        experiment_name = os.getenv('EXPERIMENT_NAME', 'fraud_detection_gnn')
        
        return {
            'explanation_output_dir': output_dir,
            'output_dir': output_dir,
            'experiment_name': experiment_name,
            'full_path': os.path.join(output_dir, experiment_name)
        }
    
    @classmethod
    def validate_datasets(cls):
        """Validate that all required datasets exist."""
        cls.load_env()
        
        datasets = {
            'financial': 'FINANCIAL_FRAUD_DATASET_PATH',
            'credit': 'GERMAN_CREDIT_DATASET_PATH',
            'loans': 'CREDIT_RISK_DATASET_PATH',
            'transaction': 'TRANSACTION_DATASET_PATH'
        }
        
        all_valid = True
        for name, env_key in datasets.items():
            path = os.getenv(env_key)
            if path and os.path.exists(path):
                print(f"✓ Found {name:12}: {path}")
            else:
                print(f"✗ Missing {name:12}: {path}")
                all_valid = False
        
        if not all_valid:
            raise FileNotFoundError("Some datasets are missing. Check paths in .env file.")
        
        return all_valid
