import argparse


class Configs(object):
    def __init__(self):
        self.config = argparse.ArgumentParser()
        self.config.add_argument('--dataset_name',default='BUSBRA')
        self.config.add_argument('--num_class',type=int,default='2')
        self.config.add_argument('--checkpoint_dir',default='./checkpoints')
        self.config.add_argument('--result_path',default='./seg_results')
        self.config.add_argument('--superimposed_path',default='./seg_results_superimposed')
        self.config.add_argument('--m_name', type=str, default='EfficientFeedbackNetwork', help='Model name')
        self.config.add_argument('--wandb',default=True,action=argparse.BooleanOptionalAction)
        self.config.add_argument('--device',type=str,default='cuda')
        self.config.add_argument('--parallel',default=True,action=argparse.BooleanOptionalAction)

        self.config.add_argument('--feedback',default=True,action=argparse.BooleanOptionalAction)
        self.config.add_argument('--train', default=True, action=argparse.BooleanOptionalAction)
        self.config.add_argument('--test', default=True,action=argparse.BooleanOptionalAction)
        self.config.add_argument('--generate',default=True,action=argparse.BooleanOptionalAction)
        self.config.add_argument('--superimposed',default=False,action=argparse.BooleanOptionalAction)
        self.config.add_argument('--generate_each_epoch',default=True,action=argparse.BooleanOptionalAction)
        
        self.config.add_argument('--BATCH_SIZE',type=int,default=4)
        self.config.add_argument('--NUM_WORKERS',type=int,default=4)
        self.config.add_argument('--lr',type=float,default=1e-4)
        self.config.add_argument('--epochs',type=int,default=50)
        self.config.add_argument('--iw',type=int,default=512)
        self.config.add_argument('--ih',type=int,default=512)
        self.config.add_argument('--vegetation',default=True,action=argparse.BooleanOptionalAction)
        self.config.add_argument('--fb_init',default=True,action=argparse.BooleanOptionalAction)
        self.config.add_argument('--feedback_type',default='ones')
        self.config.add_argument('--fb_tr_type',type=int, default=0,help='0=last, 1=all')
        self.config.add_argument('--wloss',default=True,action=argparse.BooleanOptionalAction)

        self.config.add_argument('--model',type=int,default=0)

        self.config.add_argument('--out_domain',default='CVC-ColonDB')
        self.config.add_argument('--in_domain',default='KVASIR')

        #ablation
        self.config.add_argument('--ablation',default=0,type=int)
        self.config.add_argument('--ablation_type',default='loss')
        self.config.add_argument('--abt',type=int,default=0)
    
    def parse(self, args=''):
        if args=='':
            config = self.config.parse_args()
        else:
            config = self.config.parse_args(args)
        return config