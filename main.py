import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import os
from skimage import img_as_float
class DenoisingDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (256, 256))  # Resize to 256x256

        # Normalize image to range [0, 1]
        image = img_as_float(image)

        # Add Gaussian noise to the image
        noise = np.random.normal(0, 0.1, image.shape)
        noisy_image = np.clip(image + noise, 0.0, 1.0)  # Add noise

        if self.transform:
            image = self.transform(image)
            noisy_image = self.transform(noisy_image)

        return torch.tensor(noisy_image).float().unsqueeze(0), torch.tensor(image).float().unsqueeze(0)

# Load image paths (replace with your dataset paths)
image_paths = [os.path.join('./data/', img) for img in os.listdir('./data/')]

# Apply transformations if needed
transform = transforms.Compose([transforms.ToTensor()])

# Create dataset and dataloader
dataset = DenoisingDataset(image_paths, transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()

        # Encoder: Downsampling with convolution
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 256x256 -> 128x128

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 128x128 -> 64x64

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64x64 -> 32x32
        )

        # Decoder: Upsampling with transposed convolution
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 32x32 -> 64x64
            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 64x64 -> 128x128
            nn.ReLU(),

            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 128x128 -> 256x256
            nn.Sigmoid()  # Output image in range [0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
def train_model(model, dataloader, num_epochs=10, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for noisy_imgs, clean_imgs in dataloader:
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(noisy_imgs)

            # Calculate loss
            loss = criterion(outputs, clean_imgs)
            running_loss += loss.item()

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

        epoch_loss = running_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    
    print("Training complete!")

# Initialize the model and start training
model = DenoisingAutoencoder()
train_model(model, dataloader)

def denoise_image(model, image_path):
    # Load and preprocess the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (256, 256))
    image = img_as_float(image)
    noisy_image = np.clip(image + np.random.normal(0, 0.1, image.shape), 0.0, 1.0)

    # Convert to tensor
    noisy_image_tensor = torch.tensor(noisy_image).float().unsqueeze(0).unsqueeze(0)

    # Denoise the image
    model.eval()
    with torch.no_grad():
        denoised_image = model(noisy_image_tensor).squeeze().numpy()

    # Display results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')

    plt.subplot(1, 3, 2)
    plt.title("Noisy Image")
    plt.imshow(noisy_image, cmap='gray')

    plt.subplot(1, 3, 3)
    plt.title("Denoised Image")
    plt.imshow(denoised_image, cmap='gray')
    plt.show()

# Test the denoising model on a sample image
denoise_image(model, './data/sample_image.png')

