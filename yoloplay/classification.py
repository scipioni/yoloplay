import torch
import torch.nn as nn
import os
import sys
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import csv
import argparse
from typing import List, Tuple, Optional


class NeuralNet(nn.Module):
    def __init__(self, input_size=17, hidden_size=256, num_classes=1):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
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
) -> None:
    """
    Train the pose classification model.

    Args:
        csv_file: CSV file with keypoints and labels
        model_path: Path to save the trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        device: Device to train on ('auto', 'cpu', or 'cuda')
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

    # Create data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = NeuralNet(
        input_size=34, hidden_size=256, num_classes=1
    )  # 17 keypoints * 2 coords = 34
    model.to(device_torch)

    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()  # Binary classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0

        for keypoints_batch, labels_batch in dataloader:
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
            epoch_loss += loss.item()
            predicted = (torch.sigmoid(outputs.squeeze()) > 0.5).float()
            correct += (predicted == labels_batch).sum().item()
            total += labels_batch.size(0)

        # Print epoch statistics
        avg_loss = epoch_loss / len(dataloader)
        accuracy = correct / total
        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}"
        )

    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Save the trained model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


def main():
    parser = argparse.ArgumentParser(description="Train pose classification model")
    parser.add_argument(
        "--csv", type=str, required=True, help="CSV file with keypoints and labels"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="classification.pt",
        help="Path to save the trained model",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.001, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to train on ('auto', 'cpu', or 'cuda')",
    )

    args = parser.parse_args()

    # Train the model
    train_model(
        csv_file=args.csv,
        model_path=args.model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
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
