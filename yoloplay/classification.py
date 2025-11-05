import torch
import torch.nn as nn
import os
import sys
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import cv2
import csv
import argparse
from typing import List, Tuple, Optional
from sklearn.model_selection import train_test_split


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience=10, min_delta=0.0, mode='min'):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss (lower is better), 'max' for accuracy (higher is better)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_metric):
        """
        Check if early stopping should be triggered.
        
        Args:
            val_metric: Current validation metric value
            
        Returns:
            bool: True if early stopping should be triggered
        """
        score = -val_metric if self.mode == 'min' else val_metric
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop


class NeuralNet(nn.Module):
    def __init__(self, input_size=17, hidden_size=256, num_classes=1, dropout_rate=0.5):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.l2(out)
        return out


class KeypointDataset(Dataset):
    """
    Dataset for loading keypoints and labels from CSV file or image directories.
    """

    def __init__(
        self,
        stand_dir: Optional[str] = None,
        fallen_dir: Optional[str] = None,
        detector=None,
        csv_file: Optional[str] = None,
    ):
        """
        Initialize dataset.

        Args:
            stand_dir: Directory containing standing pose images
            fallen_dir: Directory containing fallen pose images
            detector: Pose detector for keypoint extraction
            csv_file: CSV file with keypoints and labels (format: x1,y1,...,x17,y17,label)
        """
        self.data = []

        if csv_file is not None:
            self._load_from_csv(csv_file)

    def _load_from_csv(self, csv_file: str) -> None:
        """
        Load keypoints and labels from CSV file.

        Args:
            csv_file: Path to CSV file with format x1,y1,x2,y2,...,x17,y17,label
        """
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file {csv_file} does not exist")

        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 35:  # 34 coordinates + 1 label
                    print(f"Warning: Invalid row length {len(row)}, expected 35")
                    continue
                try:
                    keypoints = [float(x) for x in row[1:]]
                    label = int(row[0])
                    self.data.append((keypoints, label))
                except ValueError as e:
                    print(f"Warning: Could not parse row: {e}")

        print(f"Loaded {len(self.data)} samples from CSV")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        keypoints, label = self.data[idx]
        return torch.tensor(keypoints, dtype=torch.float32), torch.tensor(
            label, dtype=torch.float32
        )


def train_model(
    csv_file: Optional[str] = None,
    model_path: str = "models/pose_classification.pt",
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 0.001,
    device: str = "auto",
    split_ratio: float = 0.8,
    patience: int = 10,
    random_seed: int = 42,
    dropout_rate: float = 0.5,
    save_best_only: bool = False,
) -> None:
    """
    Train the pose classification model with validation and early stopping.

    Args:
        csv_file: CSV file with keypoints and labels
        model_path: Path to save the trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        device: Device to train on ('auto', 'cpu', or 'cuda')
        split_ratio: Train/validation split ratio (default: 0.8)
        patience: Early stopping patience in epochs (default: 10)
        random_seed: Random seed for reproducibility (default: 42)
        dropout_rate: Dropout rate for regularization (default: 0.5)
        save_best_only: Only save best model, not final (default: False)
    """
    # Set device
    if device == "auto":
        device_torch = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_torch = torch.device(device)

    print(f"Training on device: {device_torch}")

    if csv_file is not None:
        # Load from CSV
        dataset = KeypointDataset(csv_file=csv_file)
    
    if len(dataset) == 0:
        raise ValueError("No valid training data found")

    print(f"Loaded {len(dataset)} samples from CSV")

    # Extract labels for stratified splitting
    labels = [dataset[i][1].item() for i in range(len(dataset))]
    
    # Split dataset into train and validation sets
    train_indices, val_indices = train_test_split(
        range(len(dataset)),
        test_size=1 - split_ratio,
        random_state=random_seed,
        stratify=labels
    )
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    print(f"Train set: {len(train_dataset)} samples, Val set: {len(val_dataset)} samples")
    
    if len(val_dataset) < 10:
        print("Warning: Validation set is very small (< 10 samples). Consider using more data.")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model = NeuralNet(
        input_size=34, hidden_size=256, num_classes=1, dropout_rate=dropout_rate
    )  # 17 keypoints * 2 coords = 34
    model.to(device_torch)

    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()  # Binary classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=patience, mode='min')
    
    # Best model tracking
    best_val_loss = float('inf')
    best_model_path = model_path.replace('.pt', '_best.pt')

    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for keypoints_batch, labels_batch in train_loader:
            keypoints_batch = keypoints_batch.to(device_torch)
            labels_batch = labels_batch.to(device_torch)

            # Forward pass
            outputs = model(keypoints_batch)
            loss = criterion(outputs.squeeze(), labels_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Statistics
            train_loss += loss.item()
            predicted = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
            train_correct += (predicted == labels_batch).sum().item()
            train_total += labels_batch.size(0)

        # Calculate training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for keypoints_batch, labels_batch in val_loader:
                keypoints_batch = keypoints_batch.to(device_torch)
                labels_batch = labels_batch.to(device_torch)

                # Forward pass
                outputs = model(keypoints_batch)
                loss = criterion(outputs.squeeze(), labels_batch)

                # Statistics
                val_loss += loss.item()
                predicted = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
                val_correct += (predicted == labels_batch).sum().item()
                val_total += labels_batch.size(0)

        # Calculate validation metrics
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total

        # Print epoch statistics
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"  Train - Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.4f}")
        print(f"  Val   - Loss: {avg_val_loss:.4f}, Acc: {val_accuracy:.4f}")

        # Model checkpointing - save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(os.path.dirname(model_path) if os.path.dirname(model_path) else '.', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_acc': train_accuracy,
                'val_acc': val_accuracy,
            }, best_model_path)
            print(f"  ✓ Best model saved (val_loss: {avg_val_loss:.4f})")

        # Early stopping check
        if early_stopping(avg_val_loss):
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            print(f"Best validation loss: {best_val_loss:.4f}")
            break

    # Load best model for final save
    checkpoint = torch.load(best_model_path, map_location=device_torch)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"\nTraining completed!")
    print(f"Best model (epoch {checkpoint['epoch'] + 1}) saved to {best_model_path}")
    print(f"  Train - Loss: {checkpoint['train_loss']:.4f}, Acc: {checkpoint['train_acc']:.4f}")
    print(f"  Val   - Loss: {checkpoint['val_loss']:.4f}, Acc: {checkpoint['val_acc']:.4f}")

    # Save final model if requested
    if not save_best_only:
        torch.save(model.state_dict(), model_path)
        print(f"Final model also saved to {model_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train pose classification model with validation and early stopping"
    )
    parser.add_argument(
        "--csv", type=str, required=True, help="CSV file with keypoints and labels"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="classification.pt",
        help="Path to save the trained model (default: classification.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Maximum number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for training (default: 16)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate for optimizer (default: 0.001)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to train on: 'auto', 'cpu', or 'cuda' (default: auto)",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.8,
        help="Train/validation split ratio (default: 0.8 = 80%% train, 20%% val)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience in epochs (default: 10)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout rate for regularization (default: 0.5)",
    )
    parser.add_argument(
        "--save-best-only",
        action="store_true",
        default=True,
        help="Only save the best model, not the final model (default: True)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not 0 < args.split_ratio < 1:
        parser.error("--split-ratio must be between 0 and 1")
    if not 0 <= args.dropout < 1:
        parser.error("--dropout must be between 0 and 1")
    if args.patience < 1:
        parser.error("--patience must be at least 1")

    # Train the model
    train_model(
        csv_file=args.csv,
        model_path=args.model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        split_ratio=args.split_ratio,
        patience=args.patience,
        random_seed=args.random_seed,
        dropout_rate=args.dropout,
        save_best_only=args.save_best_only,
    )


# class KeypointClassification:
#     def __init__(self, path_model):
#         self.path_model = path_model
#         self.classes = ['Stand']
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.load_model()

#     def load_model(self):
#         self.model = NeuralNet()
#         self.model.load_state_dict(
#             torch.load(self.path_model, map_location=self.device)
#         )
#     def __call__(self, input_keypoint):
#         if not type(input_keypoint) == torch.Tensor:
#             input_keypoint = torch.tensor(
#                 input_keypoint, dtype=torch.float32
#             )
#         out = self.model(input_keypoint)
#         _, predict = torch.max(out, -1)
#         label_predict = self.classes[predict]
#         return label_predict

# if __name__ == '__main__':
#     keypoint_classification = KeypointClassification(
#         path_model='/Users/alimustofa/Me/source-code/AI/YoloV8_Pose_Classification/models/pose_classification.pt'
#     )
#     dummy_input = torch.randn(23)
#     classification = keypoint_classification(dummy_input)
#     print(classification)
