import torch
from torch.utils.data import Dataset

class DatasetMaskAugmented(Dataset):
    
    def __init__(self, input_dataset, mask,):
        super().__init__()
        self.input_dataset = input_dataset

        assert isinstance(self.input_dataset, Dataset)
        assert isinstance(mask, torch.Tensor)

        self.mask = mask
        assert self.mask.shape[0] == len(self.input_dataset)

    def __getitem__(self, idx):

        input_tensor, target = self.input_dataset.__getitem__(idx)
        mask = self.mask[idx]
        sample = {'data': input_tensor, 'target': target, 'mask': mask}
        return sample

    def get_dim_input(self):
        return next(iter(self.input_dataset))['data'].shape


    def __len__(self):
        return len(self.input_dataset)
