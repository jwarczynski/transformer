import os
import logging
import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--create_dirs', '-cds', action='store_true', default=True,
                        help='Create directories if they do not exist')
    parser.add_argument('--checkpoint_dir', '-cd', type=dir_path, default='checkpoints',
                        help='Directory to save and load checkpoints')
    parser.add_argument('--log-dir', '-ld', type=dir_path, default='logs', help='Directory to save logs')

    parser.add_argument('--dataset_train_size', '-dts', type=int_or_none, default=None,
                        help='Size of the training dataset')
    parser.add_argument('--dataset_valid_size', '-dvs', type=int_or_none, default=None,
                        help='Size of the validation dataset')

    parser.add_argument('--train_batch_size', '-tbs', type=int, default=16, help='Batch size for training')
    parser.add_argument('--valid_batch_size', '-vbs', type=int, default=16, help='Batch size for validation')
    parser.add_argument('--gradient_accumulation_steps', '-gas', type=int, default=8,
                        help='Number of steps to accumulate gradients')
    parser.add_argument('--save_checkpoint_steps', '-scs', type=int, default=8*100,
                        help='Number of steps to save a checkpoint')

    parser.add_argument('--max_train_steps', '-mts', type=int, default=-1,
                        help='Maximum number of training steps (-1 for unlimited)')
    parser.add_argument('--max_eval_steps', '-mes', type=int, default=-1,
                        help='Maximum number of evaluation steps (-1 for unlimited)')

    parser.add_argument('--project_name', '-pn', type=str, default='transformer-accelerate', help='Wandb project name')
    parser.add_argument('--run_name', '-rn', type=str_or_none, default='night training', help='Wandb run name')
    parser.add_argument('--run_id', '-ri', type=str_or_none, default=None, help='Wandb run id')
    parser.add_argument('--run_resume', '-rr', type=str_or_none, default='allow', help='Wandb run resume')

    parser.add_argument('--tokenizer-path', '-tp', type=str, default='bert-base-uncased', help='Path to the tokenizer')

    return parser.parse_args()


def int_or_none(value):
    if value == '' or value is None:
        return None
    return int(value)


def str_or_none(value):
    if value == '' or value is None:
        return None
    return value


def dir_path(value, create=False):
    if os.path.exists(value):
        return value
    if create:
        os.makedirs(value, exist_ok=True)
        return value
    raise argparse.ArgumentTypeError(f"Directory {value} does not exist")


def get_logger(log_dir=None, filename='training', stream=True, time_in_filename=True):
    if not os.path.exists(log_dir) and log_dir is not None:
        os.makedirs(log_dir)

    # log_dir = Path(log_dir) if log_dir is not None else None

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    logger.handlers = []
    # logger.handlers = [
    #     logging.FileHandler(log_dir / f"{filename}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log" if time_in_filename
    #                         else log_dir / f"{filename}.log", mode='w', encoding='utf-8')
    # ]

    if stream:
        logger.handlers.append(logging.StreamHandler())

    for handler in logger.handlers:
        handler.setFormatter(formatter)

    # logger.info(f'Logger initialized. Log directory: {log_dir}')
    logger.info(f'Logger initialized.')
    return logger
