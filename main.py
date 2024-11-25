from configs import Configs
from kvasir_loader import KVASIR_loader
from cvc_clinicdb_loader import CVC_ClinicDB_loader
from cvc300_loader import CVC300_loader
from busbra_loader import BUSBRA_loader
from isic_loader import ISIC_loader
from get_models import getModels

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from trainer import Trainer

if __name__=='__main__':
    print(1)
    config = Configs().parse()

    if config.dataset_name=='KVASIR-SEG':
        config.in_channels=3
        train_data = KVASIR_loader(augmentation=True,iw=config.iw, ih=config.ih)
        val_data = KVASIR_loader(phase='val',iw=config.iw, ih=config.ih)
        test_data = val_data
    elif config.dataset_name=='CVC-ClinicDB':
        config.in_channels=3
        train_data = CVC_ClinicDB_loader(augmentation=True,iw=config.iw, ih=config.ih)
        val_data = CVC_ClinicDB_loader(phase='val',iw=config.iw, ih=config.ih)
        test_data = val_data
    elif config.dataset_name=='CVC300':
        config.in_channels=3
        train_data = CVC300_loader(augmentation=True,iw=config.iw, ih=config.ih)
        val_data = CVC300_loader(phase='val',iw=config.iw, ih=config.ih)
        test_data = val_data

    elif config.dataset_name=='BUSBRA':
        config.num_class=2
        config.in_channels=3
        train_data = BUSBRA_loader(augmentation=True,iw=config.iw, ih=config.ih)
        val_data = BUSBRA_loader(phase='val',iw=config.iw, ih=config.ih)
        test_data = val_data
    elif config.dataset_name=='ISIC':
        config.num_class=2
        config.in_channels=3
        train_data = ISIC_loader(augmentation=True,iw=config.iw, ih=config.ih)
        val_data = ISIC_loader(phase='val',iw=config.iw, ih=config.ih)
        test_data = ISIC_loader(phase='test',iw=config.iw, ih=config.ih)

    train_loader = DataLoader(train_data,batch_size=config.BATCH_SIZE,shuffle=True,num_workers=config.NUM_WORKERS,drop_last=True)
    val_loader = DataLoader(val_data,batch_size=config.BATCH_SIZE,num_workers=config.NUM_WORKERS,shuffle=False,drop_last=True)
    test_loader = DataLoader(test_data,batch_size=1,num_workers=config.NUM_WORKERS,shuffle=False,drop_last=True)

    model = getModels(config=config)
    model.to('cuda:0')

    #make trainer
    if config.num_class==2:
        weight = torch.ones(config.num_class).to(config.device)
        weight[0] = 0.3
        weight[1] = 0.7
    else:
        weight = torch.ones(config.num_class).to(config.device)
        weight[0] = 0.3
        weight[1:] = 0.7

    config.weight = weight

    if config.wloss:
        criterion = nn.CrossEntropyLoss(weight=weight)
    else:
        criterion = nn.CrossEntropyLoss()
    
    trainer = Trainer(
        model=model,
        config=config,
        criterion=criterion,
        optimizer=optim.AdamW(model.parameters(),lr=config.lr),
    )

    if config.train:
        trainer.fit(train_loader=train_loader,val_loader=val_loader)
    
    if config.test:
        trainer.evaluator_medpy(test_loader=test_loader)
