import os
import torch

from utils.color_mask import yellow_mask

class CWL1Loss(torch.nn.Module):
    def __init__(self, weights=(2.0, 2.0, 0.2)):
        super(CWL1Loss, self).__init__()
        self.register_buffer("weights", torch.tensor(weights).view(1, 3, 1, 1))
        
    def forward(self, input, target):
        diff = torch.abs(input - target)
        loss = diff * self.weights
        return loss.mean()

class GANTrainer:
    def __init__(self, generator, discriminator, dataloader, config):
        self.G = generator
        self.D = discriminator  
        self.loader = dataloader
        self.config = config
        self.D_optimizer = torch.optim.Adam(self.D.parameters(), lr=config['D_lr'], betas=(0.5, 0.999))
        self.G_optimizer = torch.optim.Adam(self.G.parameters(), lr=config['G_lr'], betas=(0.5, 0.999))
        self.D_loss = torch.nn.MSELoss()
        self.G_loss = CWL1Loss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.G.to(self.device)
        self.D.to(self.device)
        self.D_loss.to(self.device)
        self.G_loss.to(self.device)
        self.showlog = config.get("showlog", 0)
    
    def train_D_step(self, images, real_labels):
        self.D_optimizer.zero_grad()
        pred = self.D.forward(images)
        loss = self.D_loss(pred, real_labels)
        loss.backward()
        self.D_optimizer.step()
        return loss.item()
    
    def train_D(self, num_epochs):
        for epoch in range(num_epochs):
            for batch_idx, batch in enumerate(self.loader):
                images = batch["image"]
                labels = batch["label"]
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                loss_D = self.train_D_step(images, labels)
                
                # Log the loss
            if self.showlog != 0 and epoch % self.showlog == 0:    
                print(f"Epoch [{epoch}/{num_epochs}], Loss G: {loss_D:.4f}")
    
    def train_G_step(self, null_images, labels , real_images):
        self.G_optimizer.zero_grad()
        images = self.G(null_images, labels)
        loss = self.G_loss(images, real_images)
        loss.backward()
        self.G_optimizer.step()
        return loss.item()
    
    def train_G_step_un(self, null_images, labels):
        self.G_optimizer.zero_grad()
        images = self.G(null_images, labels)
        pred = self.D.forward(images)
        loss = self.D_loss(pred, labels)
        loss.backward()
        self.G_optimizer.step()
        return loss.item()
    
    def train_G(self, num_epochs):
        for epoch in range(num_epochs):
            for batch_idx, batch in enumerate(self.loader):
                null_images = batch["null_image"]
                images = batch["image"]
                labels = batch["label"]
                null_images = null_images.to(self.device)
                images = images.to(self.device)
                labels = labels.to(self.device)
                # images = images * yellow_mask(images)
                loss_G = self.train_G_step(null_images, labels, images)
                
                # Log the loss
            if self.showlog != 0 and epoch % self.showlog == 0:    
                print(f"Epoch [{epoch}/{num_epochs}], Loss G: {loss_G:.4f}")
                    
    def save_model(self, path):
        # mkdir if not exists
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.G.state_dict(), path + "/generator.pth")
        torch.save(self.D.state_dict(), path + "/discriminator.pth")
    
    def load_model(self, path):
        self.G.load_state_dict(torch.load(path + "/generator.pth"))
        self.D.load_state_dict(torch.load(path + "/discriminator.pth"))
