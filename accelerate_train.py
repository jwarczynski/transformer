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


def setup_logging(project_name, log_file=None):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file) if log_file else logging.NullHandler()],
    )
    logger = logging.getLogger(__name__)

    if accelerator.is_main_process:
        wandb.init(project=project_name, config=args, name='night training')
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
        loss = loss.repeat(args.valid_batch_size)
        losses.append(accelerator.gather(loss))
        if 0 < args.max_eval_steps <= step:
            break

    loss = torch.mean(torch.cat(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = torch.tensor(float('inf'))

    return loss.item(), perplexity.item()


def save_checkpoint(model, optimizer, scheduler, completed_steps, path=None):
    path = f'checkpoints/checkpoint-{completed_steps}.pt' if path is None else path
    if accelerator.is_main_process:
        torch.save({
            'step': completed_steps,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, path)


if __name__ == "__main__":
    wandb.login()
    accelerator = Accelerator()
    tokenizer_path = 'bert-base-uncased'
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
        "train_batch_size": 16,
        "valid_batch_size": 16,
        "weight_decay": 0.1,
        "shuffle_buffer": 1000,
        "learning_rate": transformer_config.emb_size ** (-0.5),
        "lr_scheduler_type": "cosine",
        "num_warmup_steps": 750,
        "gradient_accumulation_steps": 8,
        "max_train_steps": -1,
        "max_eval_steps": -1,
        "seq_length": 512,
        "seed": 1,
        "save_checkpoint_steps": 16 * 8 * 100,
        "train_size": None,
        "valid_size": None,
    }
    args = Namespace(**config)

    datamanager = DataPreprocessing(
        tokenizer,
        batch_size=args.train_batch_size,
        max_length=512,
        dataset_kwargs={'path': 'wmt14', 'name': 'de-en'},
        kwargs={}
    )

    train_dataloader = get_dataloader(datamanager, 'train', size=args.train_size)
    val_dataloader = get_dataloader(datamanager, 'validation', size=args.valid_size)

    set_seed(args.seed)

    samples_per_step = accelerator.state.num_processes * args.train_batch_size

    st = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    logger, run_name = setup_logging("transformer-accelerate", f"logs/transformer-accelerate-{st}.log")
    logger.info(accelerator.state)

    model = get_model(transformer_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=transformer_config.emb_size ** (-0.5), betas=(0.9, 0.98),
                                  eps=1e-9)
    scheduler = get_lr_scheduler(optimizer, args.num_warmup_steps)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='mean')

    model, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader
    )

    from tqdm import tqdm

    model.train()
    completed_steps = 0

    # p_bar = tqdm(
    #     train_dataloader,
    #     # desc=f'Step: {step}, completed_steps: {completed_steps}, loss: {loss}, lr: {scheduler.get_last_lr()}'
    # )

    for step, batch in enumerate(train_dataloader, start=1):
        src = batch['input_ids_en']
        trg = batch['input_ids_de'][:, :-1]
        labels = batch['input_ids_de'][:, 1:]

        logits = model(src, trg)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        loss = loss / args.gradient_accumulation_steps
        accelerator.backward(loss)

        log_metrics(step, {'lr': get_lr(), 'samples': samples_per_step * step, 'steps': completed_steps,
                           'loss/train': loss.item()})

        if step % args.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            completed_steps += 1

        if completed_steps % args.save_checkpoint_steps == 0:
            logger.info('Evalating and saving model')
            eval_loss, perplexity = evaluate(model, val_dataloader, loss_fn)
            log_metrics(step, {'loss/eval': eval_loss, 'perplexity/eval': perplexity})
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            if accelerator.is_main_process:
                save_checkpoint(unwrapped_model, optimizer, scheduler, completed_steps)
            model.train()

        if completed_steps >= args.max_train_steps > 0:
            break

        # print(f'Step: {step}, completed_steps: {completed_steps}, loss: {loss}, lr: {scheduler.get_last_lr()}')
        # p_bar.set_description(f'Step: {step}, completed_steps: {completed_steps}, loss: {loss}, lr: {scheduler.get_last_lr()}')

    logger.info('Evalating and saving model after training')
    eval_loss, perplexity = evaluate(model, val_dataloader, loss_fn)
    log_metrics(step, {'loss/eval': eval_loss, 'perplexity/eval': perplexity})
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        save_checkpoint(unwrapped_model, optimizer, scheduler, completed_steps)

