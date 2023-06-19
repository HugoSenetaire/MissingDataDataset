from ..abstract_mask_generator import AbstractGenerator


class AbstractStatsGenerator(AbstractGenerator):
    """
    Abstract class for mask generators based on statistics from the full dataset. 
    As opposed to the other, the statistics are computed on the full dataset, but the mask can 
    still be generated on the fly. One requires the stats to be calculated before generating the mask.
    """

    def __init__(self, accross_channel=True):
        super().__init__(accross_channel)
        self.requires_stats = True
        self.stats_calculated = False

    def calculate_stats(self, dataset, nb_samples=10000):
        """
        Calculate the statistics on the dataset
        """
        raise NotImplementedError
    
    def masking_rule(self, batch):
        """
        Generate the mask for the batch
        """
        raise NotImplementedError
    
    def __call__(self, batch):
        if not self.stats_calculated:
            raise ValueError("Stats not calculated")
        return super().__call__(batch)