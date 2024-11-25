import cv2
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import albumentations as A
import pandas as pd

class BUSBRA_loader(Dataset):
    def __init__(self, iw=512, ih=512, augmentation=True, phase='train'):
        super().__init__()
        self.iw = iw
        self.ih = ih
        self.csv_file = "/media/mountHDD3/data_storage/biomedical_data/z2h/BUSBRA/bus_data.csv"
        self.augmentation = augmentation
        self.phase = phase
        self.num_class = 2
        self.img_normalization = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])
        self.transform_AUG =  A.Compose([
            A.Rotate(limit=35, p=0.3),
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.CoarseDropout(p=0.3, max_holes=10, max_height=32, max_width=32)
        ])
        self.mask_normalization = transforms.Compose([transforms.ToTensor()])
        self.__get_path__(phase)
    
    def __len__(self):
        return len(self.img)

    def __gen_train_test(self, split=0.8):
        df = pd.read_csv(self.csv_file)
        train_df = df.sample(frac=split, random_state=42)
        test_df = df.drop(train_df.index)

        train_img = train_df['ID'].tolist()
        self.train_img = [f'{item}.png' for item in train_img]
        self.train_mask = [f'{item}.png' for item in train_img]
        test_img = test_df['ID'].tolist()
        self.test_img = [f'{item}.png' for item in test_img]
        self.test_mask = [f'{item}.png' for item in test_img]
    
    def __get_path__(self, phase, split=0.8):
        self.__gen_train_test(split=split)
        if phase == 'train':
            self.img = self.train_img
            self.mask = self.train_mask
        else:
            self.img = self.test_img
            self.mask = self.test_mask
    
    def __getitem__(self, index):
        img_path = '/media/mountHDD3/data_storage/biomedical_data/z2h/BUSBRA/Images/'+ self.img[index]
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.iw, self.ih))
        mask_path = '/media/mountHDD3/data_storage/biomedical_data/z2h/BUSBRA/Masks/mask_' + self.mask[index].split("_")[1]
        label = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
        label = cv2.resize(label,(self.iw,self.ih),interpolation=cv2.INTER_NEAREST)  
        
        if self.augmentation:
            augmentations = self.transform_AUG(image=img, mask=label)
            img = augmentations['image']
            label = augmentations["mask"]
        
        img = self.img_normalization(img)
        label = self.mask_normalization(label)
        
        return img_path, img, label, label