import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import PreTrainedTokenizer, AutoTokenizer
import wandb
import torch.distributed as dist
import torch.multiprocessing as mp

from model import CausalTransformer
from datamanager import DataPreprocessing, get_dataloader
from utils import get_logger
import easydict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = get_logger(__name__)


def load_checkpoint(model, optimizer, scheduler, path):
    epoch = 1
    if os.path.exists(path):
        logger.info(f'Loading checkpoint from {path}')
        checkpoint = torch.load(path, map_location='cpu')

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch'] + 1

    return model, optimizer, scheduler, epoch


def get_model(config):
    model = CausalTransformer(
        config.num_encoder_layers,
        config.num_decoder_layers,
        config.n_heads,
        config.emb_size,
        config.sequence_length,
        config.src_vocab_size,
        config.trg_vocab_size,
        config.hidden_size,
        config.dropout,
    )
    return model


def get_lr_scheduler(optimizer, warmup_steps):
    def lr_lambda(step):
        step = max(1, step)
        return min(step ** -0.5, step * (warmup_steps ** -1.5))

    return LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def evaluate(model, val_dataloader, loss_fn):
    model.eval()
    total_loss = 0

    for batch in val_dataloader:
        src = batch['input_ids_en'].to(device)
        trg = batch['input_ids_de'][:, :-1].contiguous().to(device)
        labels = batch['input_ids_de'][:, 1:].contiguous().to(device)

        logits = model(src, trg)
        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        total_loss += loss.item()

    val_loss = total_loss / len(val_dataloader)
    wandb.log({'val_loss': val_loss})
    print(f'Validation Loss: {val_loss}')


def save_checkpoint(model, optimizer, scheduler, epoch, path):
    if dist.get_rank() == 0:  # Save only on the main process
        logger.info(f'Saving checkpoint to {path}')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),  # Use .module in DDP
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, path)


def train_one_epoch(model, dataloader, loss_fn, optimizer, scheduler, epoch):
    model.train()
    total_loss = 0
    current_loss = 0
    progress_bar = tqdm(
        dataloader,
        desc=f'epoch: {epoch}, total_loss: {total_loss}, current_loss: {current_loss}, lr: {optimizer.param_groups[0]["lr"]}'
    )
    for batch in progress_bar:
        optimizer.zero_grad()

        src = batch['input_ids_en'].to(device)
        trg = batch['input_ids_de'][:, :-1].to(device)
        labels = batch['input_ids_de'][:, 1:].to(device)

        logits = model(src, trg)

        loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        current_loss = loss.item()
        total_loss += current_loss
        loss.backward()

        optimizer.step()
        scheduler.step()

        progress_bar.set_description(
            f'epoch: {epoch}, total_loss: {total_loss:.4f}, current_loss: {current_loss:.4f}, lr: {optimizer.param_groups[0]["lr"]:.6f}'
        )

        wandb.log({
            'batch_loss': current_loss,
            'lr': optimizer.param_groups[0]['lr']
        })

    total_loss /= len(dataloader)
    wandb.log({'epoch_train_loss': total_loss})


def train(model, loss_fn, optimizer, scheduler, train_dataloader, val_dataloader, epochs: iter, rank, world_size):
    for epoch in epochs:
        train_one_epoch(model, train_dataloader, loss_fn, optimizer, scheduler, epoch)
        if rank == 0:  # Save checkpoint only in the main process
            evaluate(model, val_dataloader, loss_fn)
            save_checkpoint(model, optimizer, scheduler, epoch, f'checkpoint.pt')


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group('nccl', rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def main_worker(rank, world_size, transformer_config, start_epoch, session_epochs, tokenizer, train_size=None):
    setup(rank, world_size)

    datamanager = DataPreprocessing(
        tokenizer,
        batch_size=128,
        max_length=512,
        dataset_kwargs={'path': 'wmt14', 'name': 'de-en'},
        kwargs={}
    )

    train_dataloader = get_dataloader(datamanager, 'train', size=1024, world_size=world_size, rank=rank)
    val_dataloader = get_dataloader(datamanager, 'validation', size=512, world_size=world_size, rank=rank)

    model = get_model(transformer_config).to(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=transformer_config.emb_size ** -0.5, betas=(0.9, 0.98),
                                  eps=1e-9)
    scheduler = get_lr_scheduler(optimizer, 40)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='sum')

    model, optimizer, scheduler, start_epoch = load_checkpoint(model, optimizer, scheduler, 'checkpoint.pt')

    if rank == 0:  # Initialize wandb only in the main process
        wandb.init(
            project="transformer-hgx",
            config={
                'model': 'causal-transformer-fix',
                'dataset': 'wmt14',
            },
            name='causal-transformer-fix',
            notes='fixing attention',
        )
        wandb.watch(model, criterion=loss_fn, log='all', log_graph=True)

    train(model, loss_fn, optimizer, scheduler, train_dataloader, val_dataloader,
          range(start_epoch, start_epoch + session_epochs), rank, world_size)

    cleanup()


if __name__ == '__main__':
    world_size = torch.cuda.device_count()

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
    transformer_config = easydict.EasyDict(transformer_config)

    mp.spawn(main_worker,
             args=(world_size, transformer_config, 1, 2, tokenizer),
             nprocs=world_size,
             join=True)
