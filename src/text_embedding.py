import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class TextEmbeddingModel(nn.Module):
    """Text encoder with average (or CLS) pooling and L2 normalization."""

    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def pooling(self, model_output, attention_mask, use_pooling='average'):
        model_output = model_output.masked_fill(~attention_mask[..., None].bool(), 0.0)
        if use_pooling == 'average':
            emb = model_output.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        else:
            emb = model_output[:, 0]
        return emb

    def forward(self, encoded_batch):
        if 't5' in self.model_name.lower():
            input_ids = encoded_batch['input_ids']
            decoder_input_ids = torch.zeros(
                (input_ids.shape[0], 1), dtype=torch.long, device=input_ids.device)
            model_output = self.model(**encoded_batch, decoder_input_ids=decoder_input_ids)
        else:
            model_output = self.model(**encoded_batch)

        use_pooling = 'average'
        if 'bge' in self.model_name.lower() or 'mxbai' in self.model_name.lower():
            use_pooling = 'cls'
        if isinstance(model_output, tuple):
            model_output = model_output[0]
        if isinstance(model_output, dict):
            model_output = model_output['last_hidden_state']

        emb = self.pooling(model_output, encoded_batch['attention_mask'], use_pooling)
        return F.normalize(emb, dim=-1)
