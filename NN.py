import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl 
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
import numpy as np
import os


class SpectralDataset(Dataset):
    def __init__(self, data_dir, transform=False):
        self.data_dir = data_dir
        self.data_files = sorted([f for f in os.listdir(data_dir) if f.endswith('Data.txt')])
        self.para_files = sorted([f for f in os.listdir(data_dir) if f.endswith('Para.txt')])
        self.transform = transform

        self.L_data1 = 1024
        self.L_data2 = 764

        assert len(self.data_files) == len(self.para_files), "Mismatch in number of data and parameter files"

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data_path = os.path.join(self.data_dir, self.data_files[idx])
        para_path = os.path.join(self.data_dir, self.para_files[idx])

        f_x = np.loadtxt(data_path, usecols=(0), delimiter=',')
        g_x = np.loadtxt(data_path, usecols=(1), delimiter=',')

        parameters = np.loadtxt(para_path)
        
        spectra_tensor = torch.tensor(np.array([f_x, g_x]), dtype=torch.float32)
        para_tensor = torch.tensor(parameters, dtype=torch.float32)
        
        return spectra_tensor, para_tensor
    
def get_data_loaders(data_dir, batch_size, train_split=0.95, num_workers=4):

    full_dataset = SpectralDataset(data_dir)
    
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = dataset_size - train_size 
    
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,persistent_workers=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,persistent_workers=True, num_workers=num_workers)
    
    return train_loader, val_loader

class ConstrainedOutput(nn.Module):
    def __init__(self):
        super(ConstrainedOutput, self).__init__()

    def forward(self, x):
        # Constrain each output
        out1 = torch.clamp(x[:, 0].round(), min=2, max=9)
        out2 = torch.clamp(x[:, 1], min=0.1, max=1.1)
        out3 = torch.clamp(x[:, 2], min=0.1, max=1.1)
        out4 = torch.clamp(x[:, 3], min=0.1, max=1.1)
        out5 = torch.clamp(x[:, 4], min=1.2, max=1.8)

        return torch.stack([out1, out2, out3, out4, out5], dim=1)

class SmoothConstrainedOutput(nn.Module):
    def __init__(self):
        super(SmoothConstrainedOutput, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply sigmoid to map all outputs to (0, 1)
        x = self.sigmoid(x)
        
        # Scale the outputs to their respective ranges
        output = torch.zeros_like(x)
        output[:, 0] = 2 + x[:, 0] * 7  # 2 to 9
        output[:, 1] = 0.1 + x[:, 1]  # 0.1 to 1.1
        output[:, 2] = 0.1 + x[:, 2]  # 0.1 to 1.1
        output[:, 3] = 0.1 + x[:, 3]  # 0.1 to 1.1
        output[:, 4] = 1.2 + x[:, 4] * 0.6  # 1.2 to 1.8

        return output
    
class SpectralCNN(pl.LightningModule):
    def __init__(self):
        super(SpectralCNN, self).__init__()
        self.save_hyperparameters()
        # Separate convolutional processors for each spectrum
        self.spectrum_processor1 = nn.Sequential(
            nn.BatchNorm1d(1),
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten()
        )

        self.spectrum_processor2 = nn.Sequential(
            nn.BatchNorm1d(1),
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten()
        )

        self.flat_size = 32 * (1024 // 8)  

        self.joint_processor = nn.Sequential(
            nn.Linear(self.flat_size * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )

        # Constrained output layer
        self.constrained_output = ConstrainedOutput()
        self.smooth_constrained_output = SmoothConstrainedOutput()

    def forward(self, x):
        # x shape: [batch_size, 2, 1024]
        spectrum1 = x[:, 0, :].unsqueeze(1)  # [batch_size, 1, 102
        spectrum2 = x[:, 1, :].unsqueeze(1)  # [batch_size, 1, 1024]

        # Process each spectrum independently
        processed1 = self.spectrum_processor1(spectrum1)
        processed2 = self.spectrum_processor2(spectrum2)

        # Concatenate the processed spectra
        combined = torch.cat((processed1, processed2), dim=1)

        # Joint processing
        output = self.joint_processor(combined)

        # Apply constraints
        constrained_output = self.smooth_constrained_output(output)

        return constrained_output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1
            },
        }
    
    def loss_function(self, y_hat, y):
        return F.mse_loss(y_hat, y)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_function(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_function(y_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

def train_model():

    data_dir = './DataComp1'
    batch_size =64
    train_loader, val_loader = get_data_loaders(data_dir, batch_size=batch_size)
    
    #print(full_dataset.L_data2)
    model = SpectralCNN()

    trainer = pl.Trainer(max_epochs=100)
    trainer.fit(model , train_dataloaders=train_loader , val_dataloaders=val_loader)

if __name__ == '__main__':
    train_model()
