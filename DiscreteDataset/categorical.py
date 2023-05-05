import torch
from jaxtyping import Float
from torch.distributions import categorical
from torch.utils.data import Dataset


class CategoricalDataset(Dataset):
    """Torch Dataset for Categorical distributed samples

    Attributes:
        distribution: Float[torch.Tensor,  "num_categories"], tensor of probabilities for each category.
        Must sum to one.
        num_samples: int, number of samples to generate in the dataset
    """

    def __init__(self, distribution: list, num_samples: int) -> None:
        """Initialize the dataset
        Args:
            distribution: List, list of probabilities for each category.
            Must sum to one.
            num_samples: int, number of samples to generate in the dataset

        Raises:
            AssertionError: if distribution does not sum to one
        """
        self.distribution = torch.Tensor(distribution)
        assert torch.allclose(
            torch.sum(distribution), torch.tensor(1.0)
        ), f"distribution must sum to one and got {distribution}"
        self.num_samples = num_samples
        self.num_classes = len(distribution)
        self.data = categorical.Categorical(self.distribution).sample(
            (num_samples, 1, 1)
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index: int) -> Float[torch.Tensor, "1 1"]:
        return self.data[index]


class Categorical:
    """Holders for train, val and test datasets for Categorical

    Attributes:
        dataset_train: CategoricalDataset, dataset for training
        dataset_val: CategoricalDataset, dataset for validation
        dataset_test: CategoricalDataset, dataset for testing
    """

    def __init__(
        self,
        distribution: list,
        num_samples: int,
        num_samples_val: int = None,
        num_samples_test: int = None,
    ):
        """Initialize the datasets
        Args:
            distribution: List, list of probabilities for each category.
            Must sum to one.
            num_samples: int, number of samples to generate in the dataset
        """
        if num_samples_val is None:
            num_samples_val = int(num_samples * 0.1)
        if num_samples_test is None:
            num_samples_test = int(num_samples * 0.1)

        self.dataset_train = CategoricalDataset(distribution, num_samples)
        self.dataset_val = CategoricalDataset(distribution, num_samples_val)
        self.dataset_test = CategoricalDataset(distribution, num_samples_test)

    def get_dim_input(
        self,
    ):
        return (1, 1)
