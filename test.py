import argparse
import os

import pandas as pd
import torch
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

tokenizer = None


def collate_fn(batch):
    items = default_collate(batch)
    text, label, method, level, used, origin, _ids = items
    encoded = tokenizer.batch_encode_plus(
        text,
        return_tensors='pt',
        max_length=512,
        padding='max_length',
        truncation=True,
    )
    return encoded, label, method, level, used, origin, _ids


def test(opt):
    torch.set_float32_matmul_precision('medium')
    if opt.device_num > 1:
        strategy = DDPStrategy(find_unused_parameters=True)
        fabric = Fabric(accelerator='cuda', precision='bf16-mixed',
                        devices=opt.device_num, strategy=strategy)
    else:
        fabric = Fabric(accelerator='cuda', precision='bf16-mixed', devices=opt.device_num)
    fabric.launch()
    device = fabric.device

    if opt.module == 'coarse':
        model = CoarseDetector(opt, fabric)
    else:
        model = SubdivisionDetector(opt, fabric)

    checkpoint = os.path.join(opt.model_dir, 'model_classifier_best.pth')
    state_dict = torch.load(checkpoint, map_location='cpu')
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    print(f'Loaded model from {checkpoint}')
    model.eval()
    model = fabric.setup(model)

    dataset = load_MyData(opt.path)
    test_data = PassagesDataset(dataset[opt.split], need_ids=True)
    loader = DataLoader(test_data, batch_size=opt.per_gpu_eval_batch_size,
                        num_workers=opt.num_workers, shuffle=False, drop_last=False,
                        collate_fn=collate_fn)
    loader = fabric.setup_dataloaders(loader)

    all_preds, all_labels, all_scores, all_ids = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Testing', disable=fabric.global_rank != 0):
            encoded, label, method, level, used, origin, ids = batch
            encoded = {k: v.to(device) for k, v in encoded.items()}
            meta = (method.to(device), level.to(device), used.to(device), origin.to(device))
            _, out, labels = model(encoded, meta, label.to(device))
            scores = torch.softmax(out.float(), dim=1)[:, 1]
            preds = (scores > opt.threshold).long()
            ids = fabric.all_gather(ids.to(device)).view(-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_scores.extend(scores.cpu().tolist())
            all_ids.extend(ids.cpu().tolist())

    if fabric.global_rank == 0:
        m = compute_metrics(all_labels, all_preds, all_scores, all_ids)
        print('\nFinal Test Metrics:')
        print_metrics(m)
        df = pd.DataFrame({'id': all_ids, 'true_label': all_labels,
                           'pred_label': all_preds, 'score': all_scores})
        os.makedirs(os.path.dirname(opt.detail_result_path) or '.', exist_ok=True)
        df.to_csv(opt.detail_result_path, index=False)
        print(f'Saved detailed results to {opt.detail_result_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--module', type=str, default='coarse',
                        choices=['coarse', 'subdivision'])
    parser.add_argument('--device_num', type=int, default=1)
    parser.add_argument('--model_name', type=str, default='princeton-nlp/unsup-simcse-roberta-base')
    parser.add_argument('--projection_size', type=int, default=768)
    parser.add_argument('--classifier_dim', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--per_gpu_eval_batch_size', type=int, default=32)
    parser.add_argument('--path', type=str, default='data/seqxgpt')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--model_dir', type=str, default='runs/hidet_v0')
    parser.add_argument('--detail_result_path', type=str, default='results/detail_result.csv')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='prediction threshold on P(human) (Eq. 2)')
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--w_m', type=float, default=1.0)
    parser.add_argument('--w_e', type=float, default=1.0)
    parser.add_argument('--w_u', type=float, default=1.0)
    parser.add_argument('--w_oh', type=float, default=1.0)
    parser.add_argument('--w_oai', type=float, default=1.0)
    parser.add_argument('--loss_lambda', type=float, default=0.75)
    parser.add_argument('--resum', action='store_true')
    parser.add_argument('--pth_path', type=str, default='')
    opt = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(opt.model_name)
    test(opt)
