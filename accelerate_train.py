import os.path
from datetime import datetime
from accelerate import Accelerator
import logging
import wandb
from argparse import Namespace
import torch.nn.functional
import torch.nn as nn
from transformers import AutoTokenizer
import datasets
import transformers
from train import get_model, get_lr_scheduler
from accelerate.utils import set_seed

from datamanager import DataPreprocessing, get_dataloader
from utils import parse_args


def setup_logging(project_name, run_name=None, log_file=None):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file) if log_file else logging.NullHandler()],
    )
    logger = logging.getLogger(__name__)

    if accelerator.is_main_process:
        wandb.init(project=project_name, config=train_args, name=run_name, id=args.run_id, resume=args.run_resume)
        run_name = wandb.run.name
        logger.setLevel(logging.INFO)
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
    else:
        run_name = ''
        logger.setLevel(logging.ERROR)
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    return logger, run_name


def log_metrics(step, metrics):
    global logger
    logger.info(f"Step {step}: {metrics}")
    if accelerator.is_main_process:
        wandb.log(metrics)
        # [tb_writer.add_scalar(k, v, step) for k, v in metrics.items()]


def get_lr():
    return optimizer.param_groups[0]['lr']


@torch.no_grad()
def evaluate(model, val_dataloader, loss_fn):
    model.eval()
    losses = []
    for step, batch in enumerate(val_dataloader, start=1):
        src = batch['input_ids_en']
        trg = batch['input_ids_de'][:, :-1]
        labels = batch['input_ids_de'][:, 1:]

        logits = model(src, trg)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        loss = loss.repeat(train_args.valid_batch_size)
        losses.append(accelerator.gather(loss))
        if 0 < train_args.max_eval_steps <= step:
            break

    loss = torch.mean(torch.cat(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = torch.tensor(float('inf'))

    return loss.item(), perplexity.item()


def train(model, optimizer, scheduler, loss_fn, train_loader, val_loader, args):
    model.train()
    completed_steps = args.completed_steps
    samples_per_step = accelerator.state.num_processes * args.train_batch_size
    for step, batch in enumerate(train_loader, start=args.starting_step):
        src = batch['input_ids_en']
        trg = batch['input_ids_de'][:, :-1]
        labels = batch['input_ids_de'][:, 1:]

        logits = model(src, trg)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        loss = loss / args.gradient_accumulation_steps
        accelerator.backward(loss)

        log_metrics(step, {
            'lr': get_lr(),
            'samples': samples_per_step * step,
            'steps': completed_steps,
            'loss/train': loss.item(),
            'batch_tokens_enc': src.numel(),
            'batch_tokens_dec': trg.numel(),
        })

        if step % args.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            completed_steps += 1

        if step % args.save_checkpoint_steps == 0:
            logger.info('Evalating and saving model')
            eval_loss, perplexity = evaluate(model, val_loader, loss_fn)
            log_metrics(step, {
                'loss/eval': eval_loss,
                'perplexity/eval': perplexity,
                'batch_tokens_enc': src.numel(),
                'batch_tokens_dec': trg.numel(),
            })
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            if accelerator.is_main_process:
                save_checkpoint(unwrapped_model, optimizer, scheduler, step, completed_steps,
                                path=f'{train_args.save_checkpoint_dir}/checkpoint-{step}.pt')
            model.train()

        if step >= args.starting_step + args.max_train_steps > 0:
            break

    logger.info('Evalating and saving model after training')
    eval_loss, perplexity = evaluate(model, val_dataloader, loss_fn)
    log_metrics(step, {'loss/eval': eval_loss, 'perplexity/eval': perplexity})
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        save_checkpoint(
            unwrapped_model, optimizer, scheduler, step, completed_steps,
            path=f'{train_args.save_checkpoint_dir}/checkpoint-{step}.pt')


def save_checkpoint(model, optimizer, scheduler, step, completed_steps, path=None):
    path = f'checkpoints/checkpoint-{step}.pt' if path is None else path
    if accelerator.is_main_process:
        torch.save({
            'step': step,
            'completed_steps': completed_steps,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, path)


def load_checkpoint(model, optimizer, scheduler, path):
    step = 1
    completed_steps = 0
    if path is not None:
        logger.info(f'Loading checkpoint from {path}')
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        step = checkpoint['step'] + 1
        # completed_steps = checkpoint['completed_steps'] + 1
    return model, optimizer, scheduler, step, completed_steps


def find_last_checkpoint(checkpoint_dir):
    if not os.path.exists(checkpoint_dir):
        logger.warning(f'Checkpoint directory {checkpoint_dir} does not exist')
        return None
    files = os.listdir(checkpoint_dir)
    files = [f for f in files if f.startswith('checkpoint-') and f.endswith('.pt')]
    if len(files) == 0:
        logger.warning(f'No checkpoints found in {checkpoint_dir}')
        return None
    files.sort()
    last_chckpt = os.path.join(checkpoint_dir, files[-1])
    logger.info(f'Found last checkpoint at {last_chckpt}')
    return last_chckpt


if __name__ == "__main__":
    args = parse_args()

    wandb.login()

    accelerator = Accelerator()

    tokenizer_path = args.tokenizer_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    transformer_config = {
        'num_encoder_layers': 6,
        'num_decoder_layers': 6,
        'n_heads': 8,
        'emb_size': 768,
        'sequence_length': 512,
        'src_vocab_size': tokenizer.vocab_size,
        'trg_vocab_size': tokenizer.vocab_size,
        'hidden_size': 2048,
        'dropout': 0.1,
    }
    transformer_config = Namespace(**transformer_config)

    config = {
        "train_batch_size": args.train_batch_size,
        "valid_batch_size": args.valid_batch_size,
        "weight_decay": 0.1,
        "shuffle_buffer": 1000,
        "learning_rate": transformer_config.emb_size ** (-0.5),
        "lr_scheduler_type": "cosine",
        "num_warmup_steps": 750,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_train_steps": args.max_train_steps,
        "max_eval_steps": args.max_eval_steps,
        "seq_length": 512,
        "seed": 1,
        "save_checkpoint_steps": args.save_checkpoint_steps,
        "train_size": args.dataset_train_size,
        "valid_size": args.dataset_valid_size,
        "starting_step": 1
    }
    train_args = Namespace(**config)
    train_args.save_checkpoint_dir = args.checkpoint_dir

    st = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logger, run_name = setup_logging(
        "transformer-accelerate", log_file=f"logs/transformer-accelerate-{st}.log", run_name=args.run_name, )
    logger.info(accelerator.state)

    datamanager = DataPreprocessing(
        tokenizer,
        batch_size=train_args.train_batch_size,
        max_length=512,
        dataset_kwargs={'path': 'wmt14', 'name': 'de-en'},
        kwargs={}
    )

    train_dataloader = get_dataloader(datamanager, 'train', size=train_args.train_size)
    val_dataloader = get_dataloader(datamanager, 'validation', size=train_args.valid_size)

    set_seed(train_args.seed)

    model = get_model(transformer_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=transformer_config.emb_size ** (-0.5), betas=(0.9, 0.98),
                                  eps=1e-9)
    scheduler = get_lr_scheduler(optimizer, train_args.num_warmup_steps)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='mean')

    last_chckpt = find_last_checkpoint(args.checkpoint_dir)
    model, optimizer, scheduler, step, completed_steps = load_checkpoint(model, optimizer, scheduler, last_chckpt)
    train_args.starting_step = step
    train_args.completed_steps = completed_steps

    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    # Because of modification in checkpoints
    train_args.starting_step, train_args.completed_steps = 15201, 1900

    logger.info(f'Starting training from step {train_args.starting_step}')
    train(model, optimizer, scheduler, loss_fn, train_dataloader, val_dataloader, train_args)
