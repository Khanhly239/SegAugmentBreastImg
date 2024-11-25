import torch
import os

class Trainer:
    def __init__(self, model, config, criterion, optimizer):
        self.model = model
        self.config = config
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = config.device
        self.best_val_loss = float('inf')
        self.save_path = config.checkpoint_dir


    def fit(self, train_loader, val_loader):
        self.model.train()
        for epoch in range(self.config.epochs):
            running_loss = 0.0
            for batch in train_loader:
                _, inputs, labels, init_mask = batch  # Unpack the elements returned by the data loader
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                labels = labels.squeeze(1).long()  # Remove the extra dimension and convert to LongTensor
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
            
            print(f"Epoch {epoch+1}/{self.config.epochs}, Loss: {running_loss/len(train_loader)}")
            val_loss = self.validate(val_loader)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model()

    def validate(self, val_loader):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                _, inputs, labels, init_mask = batch  # Unpack the elements returned by the data loader
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                labels = labels.squeeze(1).long()  # Remove the extra dimension and convert to LongTensor
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss}")
        return val_loss

    def evaluator_medpy(self, test_loader):
        self.model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                _, inputs, labels, init_mask = batch  # Unpack the elements returned by the data loader
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                labels = labels.squeeze(1).long()  # Remove the extra dimension and convert to LongTensor
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()
        
        print(f"Test Loss: {test_loss/len(test_loader)}")

    def save_model(self):
        os.makedirs(self.save_path, exist_ok=True)
        save_file = os.path.join(self.save_path, 'model_busra.pth')
        torch.save(self.model.state_dict(), save_file)
        print(f"Model saved to {save_file}")