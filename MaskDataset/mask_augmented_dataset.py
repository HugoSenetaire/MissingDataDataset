import torch
from torch.utils.data import Dataset

class DatasetMaskAugmented(Dataset):

    def __init__(self, input_dataset, fixed_mask, generator_fixed, generator_dynamic,):
        super().__init__()
        self.input_dataset = input_dataset
        self.fixed_mask = fixed_mask
        self.generator_fixed = generator_fixed
        self.generator_dynamic = generator_dynamic

        assert isinstance(self.input_dataset, Dataset)
        assert isinstance(fixed_mask, torch.Tensor)

        self.fixed_mask = fixed_mask
        assert self.fixed_mask.shape[0] == len(self.input_dataset)
        self.len = len(self.input_dataset)

    def __getitem__(self, idx):
        # input_tensor, target = self.input_dataset.__getitem__(idx)
        dic = self.input_dataset.__getitem__(idx)
        fixed_mask = self.fixed_mask[idx]
        if self.generator_dynamic is not None :
            dynamic_mask = self.generator_dynamic({'data': dic['data'].unsqueeze(0), 'target': dic['target']})
        else :
            dynamic_mask = torch.ones_like(fixed_mask)
        mask = fixed_mask * dynamic_mask

        
        dic.update({'mask': mask, 'fixed_mask': fixed_mask, 'dynamic_mask': dynamic_mask})
        return dic

    def get_dim_input(self):
        return next(iter(self.input_dataset))['data'].shape


    def __len__(self):
        return self.len

