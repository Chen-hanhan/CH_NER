import torch
from torch.utils.data import Dataset


class NERDataset(Dataset):
    def __init__(self, task_type, features, mode, **kwargs) -> None:
        super().__init__()
        self.nums = len(features)
        # 不能为long类型
        self.token_ids = [torch.tensor(_feature.input_ids) for _feature in features]
        self.attention_masks = [torch.tensor(_feature.attention_masks) for _feature in features]
        self.token_type_ids =  [torch.tensor(_feature.token_type_ids)for _feature in features]
       
        self.labels = None
        self.start_ids, self.end_ids = None, None
        self.ent_type = None
        if mode == 'train':
            if task_type == 'crf':
                self.labels = [torch.tensor(_feature.labels).float() for _feature in features]

        #TODO: kwargs.pop('use_type_embed', False)输出什么
        if kwargs.pop('use_type_embed', False):
            self.ent_type = [torch.tensor(example.ent_type) for example in features]

    
    def __len__(self):
        return self.nums

    def __getitem__(self, index):
        data = {'token_ids': self.token_ids[index],
                'attention_masks': self.attention_masks[index],
                'token_type_ids': self.token_type_ids[index]}

        if self.ent_type is not None:
            data['ent_type'] = self.ent_type[index]

        if self.labels is not None:
            data['labels'] = self.labels[index]

        if self.start_ids is not None:
            data['start_ids'] = self.start_ids[index]
            data['end_ids'] = self.end_ids[index]

        return data

