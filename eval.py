from sklearn import metrics
import torch
import argparse
from torchvision import transforms
from PIL import Image
import os
from configs import Configs
from score2 import calculate_metrics
from torch.utils.data import DataLoader
from model6 import EfficientFeebackNetwork
from busbra_loader import BUSBRA_loader
from cvc_clinicdb_loader import CVC_ClinicDB_loader
from kvasir_loader import KVASIR_loader
from busi import BUSI_loader


def load_model(config, model_path):
    model = EfficientFeebackNetwork(num_class=config.num_class)
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    model.to(config.device)
    model.eval()
    return model

def preprocess_image(image_path, config):
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((config.ih, config.iw)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image).unsqueeze(0)  # Add batch dimension
    return image.to(config.device)

def evaluate_model(model, data_loader, config):
    model.eval()
    all_metrics = []
    with torch.no_grad():
        for batch in data_loader:
            _, inputs, labels, init_mask = batch  # Unpack the elements returned by the data loader
            inputs, labels = inputs.to(config.device), labels.to(config.device)
            labels = labels.squeeze(1).long()  # Remove the extra dimension and convert to LongTensor
            outputs = model(inputs)
            
            # Ensure y_true and y_pred have the same shape
            y_true = labels.view(-1)
            y_pred = torch.argmax(outputs, dim=1).view(-1)
            
            metrics = calculate_metrics(y_true, y_pred, config.ih, config.iw)
            all_metrics.append(metrics)
    
    avg_metrics = torch.tensor(all_metrics).mean(dim=0).tolist()
    return avg_metrics

def main(model_path, dataset_name):
    config = Configs().parse()
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.ih, config.iw = 512, 512  # Set the input image height and width

    model = load_model(config, model_path)
    
    if dataset_name == 'CVC-ClinicDB':
        test_data = CVC_ClinicDB_loader(phase='test', iw=config.iw, ih=config.ih)
    if dataset_name == 'KVASIR-SEG':
        test_data = KVASIR_loader(phase='test', iw=config.iw, ih=config.ih)
    if dataset_name == 'BUSBRA':
        test_data = BUSBRA_loader(phase='test', iw=config.iw, ih=config.ih)
    if dataset_name == 'BUSI':
        test_data = BUSI_loader(phase='test', iw=config.iw, ih=config.ih)
    # Add other datasets here if needed

    test_loader = DataLoader(test_data, batch_size=1, num_workers=config.NUM_WORKERS, shuffle=False, drop_last=True)
    
    metrics = evaluate_model(model, test_loader, config)
    print(f"IOU: {metrics[0]}")
    print(f"GIOU: {metrics[1]}")
    print(f"CIOU: {metrics[2]}")
    print(f"DIOU: {metrics[3]}")
    print(f"Dice: {metrics[4]}")
    print(f"Recall: {metrics[5]}")
    print(f"Precision: {metrics[6]}")
    print(f"F2: {metrics[-1]}")
    print(f"Evaluation Metrics: {metrics}")

if __name__ == '__main__':
    # Define the parameters directly
    model_path = '/media/mountHDD2/ly/BUS-BRA/EfficientFeedbackNetwork/checkpoints/model_busbra.pth'
    dataset_name = 'BUSBRA'

    main(model_path, dataset_name)