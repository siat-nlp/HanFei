import os
import argparse
from pathlib import Path

from src.configuration.constants import BASE_DIR

OUTPUT_DIR = f'{BASE_DIR}/output'


def add_basic_args(parser: argparse.ArgumentParser = None):
    if parser is None:
        parser = argparse.ArgumentParser()
    # directories
    parser.add_argument('--output_dir', type=str,
                        default=os.getenv('PT_OUTPUT_DIR', f'{BASE_DIR}/output/'),
                        help='directory to save logs and checkpoints to')
    parser.add_argument('--data_dir', type=str,
                        default=f'{BASE_DIR}/datasets/',
                        help='directory with train, dev, test files')

    parser.add_argument('--desc', type=str, help="Description for the experiment.")
    parser.add_argument('--seed', type=int, default=42, help="Fixing random seeds helps reproduce the result.")
    parser.add_argument('--num_epochs', type=int, default=10, help="The max training epochs.")
    parser.add_argument('--experiment_name', type=str,
                        default='plotmachine-default',
                        help='name of this experiment will be included in output')

    # single parameter
    parser.add_argument('--show_progress', action='store_true', default=True,
                        help='It will show the progress.')
    parser.add_argument('--no_gpu', action='store_true', default=False,
                        help='Runing with gpu is banned.')
    parser.add_argument('--no_multi_gpus', action='store_true', default=False,
                        help='Runing with multiple gpu is banned.')
    return parser
