# configs/__init__.py
import os
import yaml
from types import SimpleNamespace

def load_config(config_path):
    """Load and merge configurations from base and environment-specific configs"""
    # Get the directory of this configs module
    config_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load base.yaml
    base_path = os.path.join(config_dir, 'base.yaml')
    with open(base_path, 'r') as f:
        base_config = yaml.safe_load(f) or {}
    
    # Load environment-specific config
    with open(config_path, 'r') as f:
        env_config = yaml.safe_load(f) or {}
    
    # Merge configurations (env overrides base)
    config_dict = {**base_config, **env_config}
    
    # Convert to namespace object for attribute-style access
    return SimpleNamespace(**config_dict)