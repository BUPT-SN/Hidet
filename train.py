import argparse
import os
import random

import torch
import torch.optim as optim
import yaml
from lightning import Fabric
from lightning.fabric.strategies import DDPStrategy
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
from transformers import AutoTokenizer

from src.dataset import PassagesDataset
from src.models import CoarseDetector, SubdivisionDetector
from utils.dataset_utils import load_MyData
from utils.metrics import compute_metrics, print_metrics

random.seed(42)
tokenizer = None


def collate_fn(batch):
    items = default_collate(batch)
    if len(items) == 7:
        text, label, method, level, used, origin, _ids = items
    else:
        text, label, method, level, used, origin = items
    encoded = tokenizer.batch_encode_plus(
        text,
        return_tensors='pt',
        max_length=512,
        padding='max_length',
        truncation=True,
    )
    return encoded, label, method, level, used, origin


def build_fabric(opt):
    if opt.device_num > 1:
        strategy = DDPStrategy(find_unused_parameters=True)
        return Fabric(accelerator='cuda', precision='bf16-mixed',
                      devices=opt.device_num, strategy=strategy)
    return Fabric(accelerator='cuda', precision='bf16-mixed', devices=opt.device_num)


def train(opt):
    torch.set_float32_matmul_precision('medium')
    fabric = build_fabric(opt)
    fabric.launch()
    device = fabric.device

    dataset = load_MyData(opt.path)
    train_set = PassagesDataset(dataset[opt.database_name], need_ids=False)
    val_set = PassagesDataset(dataset[opt.test_dataset_name], need_ids=False)

    train_loader = DataLoader(train_set, batch_size=opt.per_gpu_batch_size,
                              num_workers=opt.num_workers, pin_memory=True,
                              shuffle=True, drop_last=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=opt.per_gpu_eval_batch_size,
                            num_workers=opt.num_workers, pin_memory=True,
                            shuffle=False, drop_last=False, collate_fn=collate_fn)

    if opt.module == 'coarse':
        model = CoarseDetector(opt, fabric).train()
    else:
        model = SubdivisionDetector(opt, fabric).train()

    if opt.freeze_embedding_layer:
        for name, param in model.model.named_parameters():
            if 'emb' in name:
                param.requires_grad = False

    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)
    num_batches = len(train_loader)

    if fabric.global_rank == 0:
        for num in range(10000):
            run_dir = os.path.join(opt.savedir, f'{opt.name}_v{num}')
            if not os.path.exists(run_dir):
                os.makedirs(run_dir)
                opt.savedir = run_dir
                break
        with open(os.path.join(opt.savedir, 'config.yaml'), 'w') as f:
            yaml.dump(vars(opt), f, sort_keys=False)
        print(f'Run directory: {opt.savedir}')

    total_steps = max(opt.total_epoch * num_batches - opt.warmup_steps, 1)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps,
                            weight_decay=opt.weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, total_steps, eta_min=opt.lr / 10)
    model, optimizer = fabric.setup(model, optimizer)

    best_f1 = 0.0
    for epoch in range(opt.total_epoch):
        model.train()
        pbar = enumerate(train_loader)
        if fabric.global_rank == 0:
            pbar = tqdm(pbar, total=num_batches)
            print(('\n%11s' * 5) % ('Epoch', 'GPU_mem', 'loss', 'avg_loss', 'train_acc'))
        right_num, tot_num, avg_loss = 0, 0, 0.0
        for i, batch in pbar:
            optimizer.zero_grad()
            step = epoch * num_batches + i
            if step < opt.warmup_steps:
                for group in optimizer.param_groups:
                    group['lr'] = opt.lr * step / opt.warmup_steps

            encoded, label, method, level, used, origin = batch
            encoded = {k: v.to(device) for k, v in encoded.items()}
            meta = (method.to(device), level.to(device), used.to(device), origin.to(device))
            loss, out, labels = model(encoded, meta, label.to(device))

            preds = torch.argmax(out, dim=1)
            right_num += (preds == labels).sum().item()
            tot_num += labels.numel()
            avg_loss = (avg_loss * i + loss.item()) / (i + 1)

            fabric.backward(loss)
            optimizer.step()
            if step >= opt.warmup_steps:
                schedule.step()

            if fabric.global_rank == 0:
                mem = f'{torch.cuda.memory_reserved() / 1e9:.3g}G'
                pbar.set_description(
                    ('%11s' * 2 + '%11.4g' * 3) %
                    (f'{epoch + 1}/{opt.total_epoch}', mem, loss.item(), avg_loss,
                     right_num / max(tot_num, 1)))

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validating',
                              disable=fabric.global_rank != 0):
                encoded, label, method, level, used, origin = batch
                encoded = {k: v.to(device) for k, v in encoded.items()}
                meta = (method.to(device), level.to(device), used.to(device), origin.to(device))
                _, out, labels = model(encoded, meta, label.to(device))
                all_preds.extend(torch.argmax(out, dim=1).cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
        torch.cuda.empty_cache()
        fabric.barrier()

        if fabric.global_rank == 0:
            m = compute_metrics(all_labels, all_preds)
            print(f'Epoch {epoch + 1} validation:')
            print_metrics(m)
            if m['f1'] > best_f1 or (epoch == opt.total_epoch - 1 and best_f1 == 0):
                best_f1 = max(best_f1, m['f1'])
                torch.save(model.get_encoder().state_dict(),
                           os.path.join(opt.savedir, 'model_best.pth'))
                torch.save(model.state_dict(),
                           os.path.join(opt.savedir, 'model_classifier_best.pth'))
                print(f'Saved best model (F1={best_f1:.4f})')
        fabric.barrier()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--module', type=str, default='coarse',
                        choices=['coarse', 'subdivision'])
    parser.add_argument('--device_num', type=int, default=1)
    parser.add_argument('--model_name', type=str, default='princeton-nlp/unsup-simcse-roberta-base')
    parser.add_argument('--projection_size', type=int, default=768)
    parser.add_argument('--classifier_dim', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--per_gpu_batch_size', type=int, default=32)
    parser.add_argument('--per_gpu_eval_batch_size', type=int, default=32)
    parser.add_argument('--path', type=str, default='data/seqxgpt')
    parser.add_argument('--database_name', type=str, default='train')
    parser.add_argument('--test_dataset_name', type=str, default='valid')
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--w_m', type=float, default=1.0)
    parser.add_argument('--w_e', type=float, default=1.0)
    parser.add_argument('--w_u', type=float, default=1.0)
    parser.add_argument('--w_oh', type=float, default=1.0)
    parser.add_argument('--w_oai', type=float, default=1.0)
    parser.add_argument('--loss_lambda', type=float, default=0.75)
    parser.add_argument('--total_epoch', type=int, default=30)
    parser.add_argument('--warmup_steps', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.98)
    parser.add_argument('--eps', type=float, default=1e-6)
    parser.add_argument('--savedir', type=str, default='./runs')
    parser.add_argument('--name', type=str, default='hidet')
    parser.add_argument('--resum', action='store_true')
    parser.add_argument('--pth_path', type=str, default='')
    parser.add_argument('--freeze_embedding_layer', action='store_true')
    opt = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(opt.model_name)
    train(opt)
