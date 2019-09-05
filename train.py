import os
import argparse
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from Dataset import BaseDataset
from Baselines import ConTagNet, UTM, USHM
from model import Net
from model_var1 import var1_net
from model_var2 import var2_net
# from ConTagNet import ConTagNet


class PHR:
    def __init__(self, args):
        ##########################################################################################################################################
        # seed = args.seed
        # np.random.seed(seed)
        # random.seed(seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        # writer = SummaryWriter()
        ##########################################################################################################################################
        self.model_name = args.model_name
        self.data_path = args.data_path
        self.learning_rate = args.l_r
        self.weight_decay = args.weight_decay
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.num_epoch = args.num_epoch
        self.dim_latent = args.dim_latent
        self.aggr_mode = args.aggr_mode
        self.num_neigh = args.num_neigh
        self.num_heads = args.num_heads
        ##########################################################################################################################################
        print('Data loading ...')
        path = '/home/share/weiyinwei/YFCC100M/'
        self.v_feat = np.load(path+'FeatureVideo_normal.npy', allow_pickle=True)
        self.a_feat = np.load(path+'FeatureAudio_avg_normal.npy', allow_pickle=True)
        self.t_feat = np.load(path+'FeatureText_normal.npy', allow_pickle=True)
        self.features = np.concatenate((self.v_feat, self.a_feat, self.t_feat), axis=1)
        ##########################################################################################################################################
        self.train_dataset = BaseDataset('/home/weiyinwei/Hashtag_recom/Data/', 'train.npy') 
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.num_user, self.num_hashtag, self.num_video = self.train_dataset.get_num()

        self.val_dataset = np.load('./Data/val.npy', allow_pickle=True)
        self.test_dataset = np.load('./Data/test.npy', allow_pickle=True)
        print('Data has been loaded.')
        ##########################################################################################################################################
        if self.model_name == 'Net':
            uh_edges_set, v_uh_edges_set = self.train_dataset.get_heter_edges()
            uh_edges_index = torch.tensor(list(uh_edges_set),dtype=torch.long).contiguous().t() 
            v_uh_edges_index = torch.tensor(list(v_uh_edges_set),dtype=torch.long).contiguous().t() 
            # v_h_edges_index = torch.tensor(list(v_h_edges_set), dtype=torch.long).contiguous().t()
            self.model = Net(self.features, uh_edges_index, v_uh_edges_index, self.batch_size, self.num_user, self.num_hashtag, self.num_video, self.dim_latent, self.num_heads, self.aggr_mode).cuda()
        elif self.model_name == 'ConTagNet':
            self.model = ConTagNet(self.features, self.batch_size, self.num_user, self.num_hashtag, self.num_video, self.dim_latent).cuda()
        elif self.model_name == 'UTM':
            self.model = UTM(self.features, self.batch_size, self.num_user, self.num_hashtag, self.num_video, self.dim_latent, 30).cuda()
        elif self.model_name == 'USHM':
            self.model = USHM(self.features, self.batch_size, self.num_user, self.num_hashtag, self.num_video, self.dim_latent).cuda()
        elif self.model_name == 'var1':
            all_edges_index = torch.tensor(list(self.train_dataset.get_edges()),dtype=torch.long).t() 
            self.model = var1_net(self.features, all_edges_index, self.batch_size, self.num_user, self.num_hashtag, self.num_video, self.dim_latent, self.aggr_mode).cuda()
        elif self.model_name == 'var2':
            uh_edges_set, v_uh_edges_set = self.train_dataset.get_heter_edges()
            uh_edges_index = torch.tensor(list(uh_edges_set),dtype=torch.long).contiguous().t() 
            v_uh_edges_index = torch.tensor(list(v_uh_edges_set),dtype=torch.long).contiguous().t() 
            # v_h_edges_index = torch.tensor(list(v_h_edges_set), dtype=torch.long).contiguous().t()
            self.model = var2_net(self.features, uh_edges_index, v_uh_edges_index, self.batch_size, self.num_user, self.num_hashtag, self.num_video, self.dim_latent, self.num_heads, self.aggr_mode).cuda()

        if args.PATH_weight_load and os.path.exists(args.PATH_weight_load):
            self.model.load_state_dict(torch.load(args.PATH_weight_load))
            print('module weights loaded....')
        # writer.add_graph(self.model, (data=data))
        ##########################################################################################################################################
        self.optimizer = torch.optim.Adam([{'params': self.model.parameters(), 'lr': self.learning_rate}], weight_decay=self.weight_decay)
        ##########################################################################################################################################

    def run(self):
        max_recall = 0.0
        # step = 0
        for epoch in range(self.num_epoch):
            self.model.train()
            print('Now, training start ...')
            pbar = tqdm(total=len(self.train_dataset))
            sum_loss = 0.0
            for data in self.train_dataloader:
                self.optimizer.zero_grad()
                self.loss = self.model.loss(data)
                # writer.add_scalar('loss', loss, step)
                # step += 1
                self.loss.backward()
                self.optimizer.step()
                pbar.update(self.batch_size)
                sum_loss += self.loss
            print(sum_loss/self.batch_size)
            pbar.close()

            print('Validation start...')
            self.model.eval()
            with torch.no_grad():
                precision, recall, ndcg_score = self.model.accuracy(self.val_dataset)

                print('---------------------------------{0}-th Precition:{1:.4f} Recall:{2:.4f} NDCG:{3:.4f}---------------------------------'.format(
                    epoch, precision, recall, ndcg_score))
                precision, recall, ndcg_score = self.model.accuracy(self.test_dataset)

                print('---------------------------------{0}-th Precition:{1:.4f} Recall:{2:.4f} NDCG:{3:.4f}---------------------------------'.format(
                    epoch, precision, recall, ndcg_score))
            if args.PATH_weight_save and recall > max_recall:
                max_recall = recall
                torch.save(self.model.state_dict(), args.PATH_weight_save)
                print('module weights saved....')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--model_name', default='Net', help='Model name.')
    parser.add_argument('--data_path', default='amazon-book', help='Dataset path')
    parser.add_argument('--PATH_weight_load', default=None, help='Loading weight filename.')
    parser.add_argument('--PATH_weight_save', default=None, help='Writing weight filename.')
    parser.add_argument('--l_r', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--dim_latent', type=int, default=64, help='Latent dimension.')
    parser.add_argument('--num_epoch', type=int, default=50, help='Epoch number.')
    parser.add_argument('--num_workers', type=int, default=40, help='Workers number.')
    parser.add_argument('--num_user', type=int, default=55485, help='User number.')
    parser.add_argument('--num_item', type=int, default=5986, help='Item number.')
    parser.add_argument('--num_neigh', type=int, default=30, help='Neighbour number.')
    parser.add_argument('--num_heads', type=int, default=1, help='Heads numbers.')
    parser.add_argument('--aggr_mode', default='mean', help='Aggregation mode.')
    parser.add_argument('--scoring_mode', default='cat', help='Scoring mode.')
    args = parser.parse_args()

    phr = PHR(args)
    phr.run()
