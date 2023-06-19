# Mask generator 
import torch

class AbstractGenerator():
    """
    Returned mask is sampled from component-wise independent Bernoulli
    distribution with probability of component to be unobserved p.
    Such mask induces the type of missingness which is called
    in literature "missing completely at random" (MCAR).
    If some value in batch is missed, it automatically becomes unobserved.
    
    If accross_channel is True, the mask is the same for all channels


    """
    def __init__(self, accross_channel = True):
        self.accross_channel = accross_channel
        self.requires_stats = False
    
    def masking_rule(self, batch):
        '''
        Given a batch, produce a mask representing the missing values the same shape as data
        data should be a tensor of shape (batch_size, num_channels, shape_x, shape_y, shape_z...)
        1 for missing values, 0 for observed values
        '''
        raise NotImplementedError

    def __call__(self, batch):
        '''
        Given a batch, produce a mask representing the missing values
        1 for missing values, 0 for observed values
        '''
        mask = self.masking_rule(batch)
        nan_mask = (1-torch.isnan(batch['data']).int())  # Missing values from the dataset
        mask = torch.min(mask, nan_mask) 
        return mask


class NoneMaskGenerator(AbstractGenerator):
    '''
    This mask generator does not mask anything
    '''

    def __init__(self, accross_channel = True):
        super().__init__(accross_channel = accross_channel)

    def masking_rule(self, batch):
        return torch.ones_like(batch['data']).int()