import torch
from jaxtyping import Float
from torch.utils.data import Dataset


class PoissonDataset(Dataset):
    """Torch Dataset for Poisson distributed samples

    Attributes:
        lambda_: float, the real valued parameter of the poisson distribution
        num_samples: int, number of samples to generate in the dataset
        data: torch.Tensor, tensor of shape (num_samples, 1, 1) containing the samples
    """

    def __init__(self, lambda_: float, num_samples: int) -> None:
        """Initialize the dataset
        Args:
            lambda_: float, the real valued parameter of the poisson distribution
            num_samples: int, number of samples to generate in the dataset
            data: torch.Tensor, tensor of shape (num_samples, 1, 1) containing the samples

        Raises:
            AssertionError: if lambda_ is not positive
        """
        self.lambda_ = lambda_
        assert lambda_ > 0, f"lambda_ must be positive and got {lambda_}"
        self.num_samples = num_samples
        self.data = torch.poisson(torch.ones(num_samples, 1, 1) * lambda_)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index: int) -> Float[torch.Tensor, "1 1"]:
        return self.data[index]


class Poisson:
    """Holders for train, val and test datasets for Poisson

    Attributes:
        dataset_train: PoissonDataset, dataset for training
        dataset_val: PoissonDataset, dataset for validation
        dataset_test: PoissonDataset, dataset for testing
    """

    def __init__(
        self,
        lambda_: float,
        num_samples: int,
        num_samples_val: int = None,
        num_samples_test: int = None,
    ):
        """Initialize the datasets
        Args:
            lambda_: float, the real valued parameter of the poisson distribution
            num_samples: int, number of samples to generate in the dataset
        """
        if num_samples_val is None:
            num_samples_val = int(num_samples * 0.1)
        if num_samples_test is None:
            num_samples_test = int(num_samples * 0.1)

        self.dataset_train = PoissonDataset(lambda_, num_samples)
        self.dataset_val = PoissonDataset(lambda_, num_samples_val)
        self.dataset_test = PoissonDataset(lambda_, num_samples_test)

    def get_dim_input(
        self,
    ):
        return (1, 1)
