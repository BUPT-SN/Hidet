import torch
import torch.nn as nn
import torch.nn.functional as F

from src.text_embedding import TextEmbeddingModel


class ClassificationHead(nn.Module):
    """3-layer MLP head, e.g. 768 -> 192 -> 48 -> 2."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.dense1 = nn.Linear(in_dim, in_dim // 4)
        self.dense2 = nn.Linear(in_dim // 4, in_dim // 16)
        self.out_proj = nn.Linear(in_dim // 16, out_dim)
        nn.init.xavier_uniform_(self.dense1.weight)
        nn.init.xavier_uniform_(self.dense2.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

    def forward(self, features):
        x = torch.tanh(self.dense1(features))
        x = torch.tanh(self.dense2(x))
        return self.out_proj(x)


def _granularity_logits(sims, pos_mask):
    """Mean positive similarity (Eq. 4) as the positive logit, each negative
    similarity as its own negative logit (Eq. 5)."""
    pos = torch.sum(sims * pos_mask, dim=1) / torch.clamp(torch.sum(pos_mask, dim=1), min=1e-6)
    neg = sims * torch.logical_not(pos_mask)
    return torch.cat([pos.unsqueeze(1), neg], dim=1)


def _masked_ce(logits, mask):
    logits = logits[mask]
    if logits.numel() == 0:
        return logits.new_zeros(())
    gt = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits.float(), gt)


class CoarseDetector(nn.Module):
    """Coarse module: encoder + classification head, trained with pure
    cross-entropy (Eq. 1). At inference, score = P(human); samples with
    score > threshold are routed to the subdivision module (Eq. 2)."""

    def __init__(self, opt, fabric):
        super().__init__()
        self.fabric = fabric
        self.model = TextEmbeddingModel(opt.model_name)
        if opt.resum:
            self.model.load_state_dict(torch.load(opt.pth_path, map_location='cpu'))
        self.classifier = ClassificationHead(opt.projection_size, opt.classifier_dim)

    def get_encoder(self):
        return self.model

    def forward(self, batch, meta, label):
        q = self.model(batch)
        out = self.classifier(q)
        loss = F.cross_entropy(out, label)
        out = self.fabric.all_gather(out).view(-1, out.size(1))
        labels = self.fabric.all_gather(label).view(-1)
        return loss, out, labels


class SubdivisionDetector(nn.Module):
    """Subdivision module trained with multi-grained contrastive learning.

    Granularity losses on AI samples: method (m), level (e), used (u) and
    origin (o-AI); on human samples: origin (o-H). Combined with the
    classification loss via L_final = lambda * L_ca + (1 - lambda) * L_ce
    (Eq. 6-7).
    """

    def __init__(self, opt, fabric):
        super().__init__()
        self.fabric = fabric
        self.temperature = opt.temperature
        self.w_m, self.w_e, self.w_u = opt.w_m, opt.w_e, opt.w_u
        self.w_oh, self.w_oai = opt.w_oh, opt.w_oai
        self.loss_lambda = opt.loss_lambda
        self.model = TextEmbeddingModel(opt.model_name)
        if opt.resum:
            self.model.load_state_dict(torch.load(opt.pth_path, map_location='cpu'))
        self.classifier = ClassificationHead(opt.projection_size, opt.classifier_dim)

    def get_encoder(self):
        return self.model

    def forward(self, batch, meta, label):
        method, level, used, origin = meta
        q = self.model(batch)
        out = self.classifier(q)

        if not self.training:
            loss = F.cross_entropy(out, label)
            out = self.fabric.all_gather(out).view(-1, out.size(1))
            labels = self.fabric.all_gather(label).view(-1)
            return loss, out, labels

        k = self.fabric.all_gather(q.clone().detach()).view(-1, q.size(1))
        k_method = self.fabric.all_gather(method).view(-1)
        k_level = self.fabric.all_gather(level).view(-1)
        k_used = self.fabric.all_gather(used).view(-1)
        k_origin = self.fabric.all_gather(origin).view(-1)

        sims = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).T / self.temperature

        same_method = method.view(-1, 1) == k_method.view(1, -1)
        same_level = level.view(-1, 1) == k_level.view(1, -1)
        same_used = used.view(-1, 1) == k_used.view(1, -1)
        same_origin = origin.view(-1, 1) == k_origin.view(1, -1)
        is_human = label == 1
        is_ai = label == 0

        loss_m = _masked_ce(_granularity_logits(sims, same_method), is_ai)
        loss_e = _masked_ce(_granularity_logits(sims, same_level), is_ai)
        loss_u = _masked_ce(_granularity_logits(sims, same_used), is_ai)
        loss_oai = _masked_ce(_granularity_logits(sims, same_origin), is_ai)
        loss_oh = _masked_ce(_granularity_logits(sims, same_origin), is_human)

        n = label.numel()
        n_human = is_human.sum().float()
        n_ai = is_ai.sum().float()
        loss_ca = (n_human / n) * self.w_oh * loss_oh + \
            (n_ai / n) * (self.w_m * loss_m + self.w_e * loss_e +
                          self.w_u * loss_u + self.w_oai * loss_oai)
        loss_ce = F.cross_entropy(out, label)
        loss = self.loss_lambda * loss_ca + (1 - self.loss_lambda) * loss_ce

        out = self.fabric.all_gather(out).view(-1, out.size(1))
        labels = self.fabric.all_gather(label).view(-1)
        return loss, out, labels
