


import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import timm
import kornia.augmentation as K
from pytorch_lightning import LightningModule


def ResNet18(in_channels, input_size):
    return timm.create_model('resnet18', pretrained=True, in_channels=in_channels, num_classes=input_size)
     
class SequenceClassificationModel(nn.Module):
    """
    Neural network model for sequence classification tasks.

    This model consists of a ResNet18 encoder, a bidirectional GRU, and a fully connected classifier.
    Given an input sequence of images, it outputs class probabilities for each sequence.

    Attributes:
    - encoder: ResNet18 encoder for feature extraction from each image in the sequence.
    - gru: Bidirectional GRU to model temporal dependencies in the sequence of features.
    - fc: Fully connected layer to produce class probabilities.
    """

    def __init__(self, in_channels, input_size, hidden_size, num_layers, num_classes):
        super(SequenceClassificationModel, self).__init__()

        self.encoder = ResNet18(in_channels, input_size)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True) # for modeling sequences

        self.fc = nn.Linear(hidden_size * 2, num_classes) # for outputting class probabilities
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
        - x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, channels, height, width).

        Returns:
        - torch.Tensor: Output tensor of class probabilities with shape (batch_size, num_classes).
        """
        batch_size, sequence_length, channels, height, width = x.shape
        x = x.view(batch_size * sequence_length, channels, height, width)
        x = self.encoder(x)
        x = x.view(batch_size, sequence_length, -1)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x
    

class SequenceAugmentationPipeline(nn.Module):
    """
    A data augmentation pipeline for sequences of images.

    This module defines a set of transformations that are applied consistently across
    all images in a sequence. This ensures that the spatial relationship between
    images in a sequence remains consistent after augmentation.

    Attributes:
    - hflip: Random horizontal flip transformation.
    - vflip: Random vertical flip transformation.
    - rotate: Random rotation transformation.
    """

    def __init__(self) -> None:
        """
        Initialize the augmentation pipeline with desired transformations.
        """
        super(SequenceAugmentationPipeline, self).__init__()

        self.hflip = K.RandomHorizontalFlip()
        self.vflip = K.RandomVerticalFlip()
        self.rotate = K.RandomRotation(degrees=30)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Apply the transformations consistently across each image in the sequence.

        Parameters:
        - input (torch.Tensor): Input tensor of shape (batch_size, sequence_length, bands, height, width).

        Returns:
        - torch.Tensor: Augmented tensor with the same shape as input.
        """
        # Extract the shape parameters for the transformations from the first image
        # in the sequence. This ensures consistent augmentation across all images.
        hflip_params = self.hflip.forward_parameters(input[:, 0, ...].shape)
        vflip_params = self.vflip.forward_parameters(input[:, 0, ...].shape)
        rotate_params = self.rotate.forward_parameters(input[:, 0, ...].shape)

        # Apply the transformations to each image in the sequence.
        transformed_seq = []
        for image in input.unbind(dim=1):
            image = self.hflip(image, hflip_params)
            image = self.vflip(image, vflip_params)
            image = self.rotate(image, rotate_params)
            transformed_seq.append(image)

        # Combine the transformed images back into the sequence format.
        output = torch.stack(transformed_seq, dim=1)

        return output
    
class SequenceClassificationTask(LightningModule):
    """
    Lightning module for the sequence classification task.

    This module wraps the SequenceClassificationModel for training, validation, and testing.
    It also handles data augmentation using the SequenceAugmentationPipeline.

    Attributes:
    - model: The sequence classification model.
    - loss_fn: Loss function for classification.
    - learning_rate: Learning rate for the optimizer.
    - aug: Data augmentation pipeline for training sequences.
    """

    def __init__(self, input_size, hidden_size, in_channels=14, num_layers=3, num_classes=7, learning_rate=0.001):
        """
        Initialize the lightning module.

        Parameters:
        - input_size (int): Size of the input to the GRU.
        - hidden_size (int): Size of the GRU hidden state.
        - in_channels (int, optional): Number of input channels to the model. Defaults to 14.
        - num_layers (int, optional): Number of GRU layers. Defaults to 3.
        - num_classes (int, optional): Number of classification classes. Defaults to 7.
        - learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.001.
        """
        super(SequenceClassificationTask, self).__init__()

        self.model = SequenceClassificationModel(in_channels, input_size, hidden_size, num_layers, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

        # Define the data augmentation pipeline for training.
        self.aug = SequenceAugmentationPipeline()

    def forward(self, x):
        """
        Forward pass through the model.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Model predictions.
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Defines a single step during training.

        Parameters:
        - batch (dict): Batch of data.
        - batch_idx (int): Index of the batch.

        Returns:
        - torch.Tensor: Training loss.
        """
        x, y = batch["image"], batch["label"]

        # Apply data augmentation to the training data.
        x = self.aug(x)

        logits = self(x)
        loss = self.loss_fn(logits, y)

        # Log training loss to TensorBoard.
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Defines a single step during validation.

        Parameters:
        - batch (dict): Batch of data.
        - batch_idx (int): Index of the batch.

        Returns:
        - torch.Tensor: Validation loss.
        """
        x, y = batch["image"], batch["label"]
        logits = self(x)
        loss = self.loss_fn(logits, y)

        # Log validation loss to TensorBoard.
        self.log("val_loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        """
        Defines a single step during testing.

        Parameters:
        - batch (dict): Batch of data.
        - batch_idx (int): Index of the batch.

        Returns:
        - torch.Tensor: Testing loss.
        """
        x, y = batch["image"], batch["label"]
        logits = self(x)
        loss = self.loss_fn(logits, y)

        # Log testing loss to TensorBoard.
        self.log("test_loss", loss)

        return loss

    def configure_optimizers(self):
        """
        Configures the optimizer(s) and learning rate scheduler(s).

        Returns:
        - Dict: Contains optimizer and learning rate scheduler information.
        """
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        # Define a learning rate scheduler that reduces the learning rate when the validation loss plateaus.
        scheduler = ReduceLROnPlateau(optimizer, patience=5)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
