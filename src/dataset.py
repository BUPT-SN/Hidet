from torch.utils.data import Dataset

from utils.dataset_utils import HUMANIZED_LEVELS, LEVEL_IDS, METHOD_INDEX


class PassagesDataset(Dataset):
    """Returns (text, label, method, level, used, origin[, id]).

    label/origin: 1 = human, 0 = AI. method: id of the humanizing method
    (or 'human'/'machine_origin'). level: 0-3 = char/word/sent/para,
    4 = human, 5 = machine_origin. used: 1 if the text is humanized AI.
    """

    def __init__(self, dataset, need_ids=True):
        self.dataset = dataset
        self.need_ids = need_ids
        self.method_index = dict(METHOD_INDEX)
        next_id = len(self.method_index)
        self.classes = list(self.method_index)
        for item in dataset:
            src = item[2]
            if src not in self.method_index:
                self.method_index[src] = (next_id, LEVEL_IDS['origin'])
                self.classes.append(src)
                next_id += 1

    def get_classes(self):
        return self.classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text, label, src, sid = self.dataset[idx]
        method, level = self.method_index[src]
        used = int(level in HUMANIZED_LEVELS)
        origin = int(label)
        if self.need_ids:
            return text, int(label), int(method), int(level), used, origin, int(sid)
        return text, int(label), int(method), int(level), used, origin
