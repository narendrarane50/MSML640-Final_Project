import yaml
import os

def load_yaml_config(config_path: str, spec: str) -> dict:
    """Load a specific configuration section from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    if spec is None:
        return config
    if spec not in config:
        raise KeyError(f"Specification '{spec}' not found in config file.")
    
    return config[spec]