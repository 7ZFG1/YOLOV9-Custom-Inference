import argparse
import yaml

def parse_args(path):
    parser = argparse.ArgumentParser(description='Inference Config')
    parser.add_argument('--config', type=str, default=path, help='Path to the config file')
    args = parser.parse_args()

    config_file = args.config

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    return config
