from pathlib import Path
from random import shuffle
from typing import Any, Callable, Dict, List, Optional
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch
import numpy as np
import matplotlib.pyplot as plt
from lightning import LightningDataModule
from torch.utils.data import DataLoader

class FieldSequenceDataset(Dataset):

    
    """
    A dataset class for sequences of field images.

    Attributes:
    - X: Numpy array containing image sequences.
    - y: Labels associated with each image sequence.
    - classes: List of class names/labels.
    - transforms: Optional data augmentation operations.

    Methods:
    - __len__ : Returns the length of the dataset.
    - __getitem__ : Fetches a data sample for a given index.
    - plot: Plots an image sequence from a given sample.
    """

    def __init__(
        self,
        X,
        y,
        field_ids: List[int],
        transforms: Optional[Callable] = None
    ) -> None:
        """
        Initializes the dataset object.

        Parameters:
        - X: Numpy array containing image sequences of shape (num_samples, num_images, height, width, bands).
        - y: Numpy array containing labels for each sequence.
        - field_ids: List of indices to subset the dataset. Defaults to None (use all data).
        - transforms: Optional data augmentation operations.
        """

        # Define class labels
        self.classes = [str(i) for i in range(1, 8)]

        # Instead of slicing the data, store the indices
        self.field_ids = field_ids
        self.X = X
        self.y = y

        # Set the data augmentation transforms
        self.transforms = transforms

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.field_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """
        Returns a data sample given an index.

        Parameters:
        - index: Index of the sample to fetch.

        Returns:
        Dictionary containing the image sequence and its associated label.
        """
        #  Use the field_ids to fetch the relevant data
        sequence = self.X[self.field_ids[index]]
        label = self.y[self.field_ids[index]]

        # Convert them to PyTorch tensors
        sample = {'image': torch.tensor(sequence, dtype=torch.float32), 'label': torch.tensor(label, dtype=torch.long)}

        return sample
    def plot(
            

        
        self,
        sample: Dict[str, Any],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plots an image sequence from a sample.

        Parameters:
        - sample: Dictionary containing an image sequence and its label.
        - show_titles: Whether to display titles on the plots.
        - suptitle: Optional overarching title for the entire plot.

        Returns:
        Matplotlib figure object.
        """

        # Extract and normalize image sequence
        sequence = sample['image'].numpy()[:, [3, 2, 1], :, :]
        label = sample['label'].item()
        min_vals = sequence.min(axis=(0, 2, 3), keepdims=True)
        max_vals = sequence.max(axis=(0, 2, 3), keepdims=True)
        sequence = (sequence - min_vals) / (max_vals - min_vals)

        # Calculate layout for plotting multiple images
        num_images = sequence.shape[0]
        num_rows = int(np.ceil(num_images / 4.0))

        # Create a figure and plot each image in the sequence
        fig, axarr = plt.subplots(num_rows, 4, figsize=(15, 4 * num_rows))
        if num_rows == 1:
            axarr = np.expand_dims(axarr, axis=0)
        for i in range(num_rows):
            for j in range(4):
                idx = i * 4 + j
                if idx < num_images:
                    ax = axarr[i, j]
                    ax.imshow(sequence[idx].transpose(1, 2, 0))
                    ax.axis('off')
                    if show_titles and idx == num_images - 1:
                        ax.set_title(f'Label: {self.classes[label]}')
                else:
                    axarr[i, j].axis('off')

        # Set the optional overarching title
        if suptitle:
            fig.suptitle(suptitle, fontsize=16)

        return fig
    
class FieldDataModule(LightningDataModule):
    """
    PyTorch Lightning data module for handling field sequence data.

    This class helps in loading and splitting the dataset into train, validation, and test sets.

    Attributes:
    - root: The path to the root directory containing the data.
    - batch_size: Size of the batches during training.
    - workers: Number of workers for data loading.
    - X: Numpy array containing image sequences.
    - y: Numpy array containing labels for each sequence.
    - train_ids, val_ids, test_ids: Lists containing indices for the train, validation, and test splits respectively.
    - train_ds, val_ds, test_ds: Dataset objects for the train, validation, and test sets.
    """

    def __init__(
        self,
        root: str,
        train_size: float = 0.8,
        val_size: float = 0.1,
        test_size: float = 0.1,
        batch_size: int = 8,
        workers: int = 4,
    ):
        super().__init__()

        # Define directory path and loading configurations
        self.root = Path(root)
        self.batch_size = batch_size
        self.workers = workers

        # Load the dataset into memory
        self.X = np.load(self.root / "X.npy")
        self.y = np.load(self.root / "y.npy")

        # Randomly shuffle field IDs for dataset split
        all_field_ids = list(range(3280))
        shuffle(all_field_ids)

        # Split the dataset into train, validation, and test sets based on provided ratios
        self.train_ids, temp_ids = train_test_split(all_field_ids, test_size=1 - train_size, random_state=42)
        self.val_ids, self.test_ids = train_test_split(temp_ids, test_size=test_size / (test_size + val_size), random_state=42)

        # Setup datasets
        self.setup()

    def setup(self, stage=None):
        """
        Prepare datasets for training, validation, and testing.

        Uses the field IDs generated during initialization to subset the full dataset.
        """
        self.train_ds = FieldSequenceDataset(self.X[self.train_ids], self.y[self.train_ids])
        self.val_ds = FieldSequenceDataset(self.X[self.val_ids], self.y[self.val_ids])
        self.test_ds = FieldSequenceDataset(self.X[self.test_ids], self.y[self.test_ids])

    def train_dataloader(self):
        """Returns a DataLoader object for the training dataset."""

        return DataLoader(self.train_ds, batch_size = self.batch_size, num_workers = self.workers, shuffle  = True)

    def val_dataloader(self):
        """Returns a DataLoader object for the validation dataset."""

        return DataLoader(self.val_ds, batch_size = self.batch_size, num_workers = self.workers)

    def test_dataloader(self):
        """Returns a DataLoader object for the test dataset."""
        return DataLoader(self.test_ds, batch_size = self.batch_size, num_workers = self.workers)