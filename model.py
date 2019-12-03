import math
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.nn.inits import uniform
# from BaseModel import BaseModel, AttenModel
# from BaseModel import SAGEConv, GATConv
from torch_geometric.nn import SAGEConv, GATConv


class Net(torch.nn.Module):
    def __init__(self, features, uh_edge_index, v_uh_edge_index, batch_size, num_user, num_hashtag, num_video, dim_latent, aggr='mean'):
        super(Net, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_hashtag = num_hashtag
        self.num_video = num_video
        self.dim_feat = features.shape[1]
        self.dim_latent = dim_latent
        self.aggr = aggr
        self.uh_edge_index = uh_edge_index.cuda()
        self.v_uh_edge_index = v_uh_edge_index.cuda()

        self.u_h_embedding = nn.init.xavier_normal_(torch.rand((num_user+num_hashtag, self.dim_latent), requires_grad=True)).cuda()
        self.video_features = torch.tensor(features, dtype=torch.float).cuda()
        self.trans_video_layer = nn.Linear(self.dim_feat, self.dim_latent)
        self.result_embed = nn.init.xavier_normal_(torch.rand((num_user+num_hashtag, self.dim_latent))).cuda()

        self.first_conv = GATConv(self.dim_latent, self.dim_latent, self.aggr)
        nn.init.xavier_normal_(self.first_conv.weight)      
        self.second_conv = SAGEConv(self.dim_latent, self.dim_latent, self.aggr)
        nn.init.xavier_normal_(self.second_conv.weight)
        self.third_conv = GATConv(self.dim_latent, self.dim_latent, self.aggr)
        nn.init.xavier_normal_(self.third_conv.weight)
        self.forth_conv = SAGEConv(self.dim_latent, self.dim_latent, self.aggr)
        nn.init.xavier_normal_(self.forth_conv.weight)

        self.user_video_layer = nn.Linear(2*self.dim_latent, self.dim_latent)
        self.user_hashtag_layer = nn.Linear(2*self.dim_latent, self.dim_latent)

    def forward(self, item):        
        user_tensor = item[:,[0]]
        video_tensor = item[:,[1]]
        pos_hashtag_tensor = item[:,[2]]
        neg_hashtag_tensor = item[:,[3]]
        x = F.leaky_relu(self.trans_video_layer(self.video_features))
        x = torch.cat((self.u_h_embedding, x), dim=0)
        x =  F.normalize(x)

        x = F.leaky_relu(self.first_conv(x, self.v_uh_edge_index))
        x = F.leaky_relu(self.second_conv(x, self.uh_edge_index))
        x = F.leaky_relu(self.third_conv(x, self.v_uh_edge_index))
        x = F.leaky_relu(self.forth_conv(x, self.uh_edge_index))
        # x = F.leaky_relu(self.first_conv(x, self.v_uh_edge_index))
        # x = F.leaky_relu(self.second_conv(x, self.uh_edge_index))
        # x = F.leaky_relu(self.first_conv(x, self.v_uh_edge_index))
        # x = F.leaky_relu(self.second_conv(x, self.uh_edge_index))
        
        self.result_embed = x[torch.arange(self.num_user+self.num_hashtag).cuda()]

        user_tensor = self.result_embed[user_tensor].squeeze(1)
        pos_hashtags_tensor = self.result_embed[pos_hashtag_tensor].squeeze(1)
        neg_hashtags_tensor = self.result_embed[neg_hashtag_tensor].squeeze(1)

        video_tensor = self.video_features[video_tensor-self.num_user-self.num_hashtag].squeeze(1)
        video_tensor = F.leaky_relu(self.trans_video_layer(video_tensor))
        user_specific_video = F.leaky_relu(self.user_video_layer(torch.cat((video_tensor, user_tensor), dim=1)))
        user_specific_pos_h = F.leaky_relu(self.user_hashtag_layer(torch.cat((pos_hashtags_tensor, user_tensor), dim=1)))
        user_specific_neg_h = F.leaky_relu(self.user_hashtag_layer(torch.cat((neg_hashtags_tensor, user_tensor), dim=1)))

        pos_scores = torch.sum(user_specific_video*user_specific_pos_h, dim=1)
        neg_scores = torch.sum(user_specific_video*user_specific_neg_h, dim=1)
        return pos_scores, neg_scores


    def loss(self, data):
        pos_scores, neg_scores = self.forward(data)
        loss_value = -torch.sum(torch.log2(torch.sigmoid(pos_scores-neg_scores)))
        return loss_value


    def accuracy(self, dataset, topk=10, neg_num=1000):
        all_set = set(list(np.arange(neg_num)))
        sum_pre = 0.0
        sum_recall = 0.0
        sum_ndcg = 0.0
        sum_item = 0
        bar = tqdm(total=len(dataset))

        for data in dataset:
            bar.update(1)
            if len(data) < 1003:
                continue

            sum_item += 1
            user = torch.tensor(data[0], dtype=torch.long)
            video = torch.tensor(data[1], dtype=torch.long)
            neg_hashtag = data[2:1002]
            pos_hashtag = data[1002:]

            pos_hashtags_tensor = torch.tensor(pos_hashtag, dtype=torch.long).cuda()
            neg_hashtags_tensor = torch.tensor(neg_hashtag, dtype=torch.long).cuda()

            user_tensor = self.result_embed[user]
            pos_hashtags_tensor = self.result_embed[pos_hashtags_tensor]
            neg_hashtags_tensor = self.result_embed[neg_hashtags_tensor]
            video_tensor = self.video_features[video-self.num_user-self.num_hashtag]
            video_tensor = F.leaky_relu(self.trans_video_layer(video_tensor))
            # user_specific_video = F.leaky_relu(self.user_video_layer(torch.cat((video_tensor, user_tensor))))
            # user_specific_pos_h = F.leaky_relu(self.user_hashtag_layer(torch.cat((pos_hashtags_tensor, user_tensor.unsqueeze(0).repeat(pos_hashtags_tensor.size(0),1)), dim=1)))
            # user_specific_neg_h = F.leaky_relu(self.user_hashtag_layer(torch.cat((neg_hashtags_tensor, user_tensor.unsqueeze(0).repeat(neg_hashtags_tensor.size(0),1)), dim=1)))
            user_specific_video = F.leaky_relu(torch.matmul(video_tensor, self.weight_v)+torch.matmul(user_tensor, self.weight_v_u)+self.bias_v)
            user_specific_pos_h = F.leaky_relu(torch.matmul(pos_hashtags_tensor, self.weight_h)+torch.matmul(user_tensor, self.weight_h_u)+self.bias_h)
            user_specific_neg_h = F.leaky_relu(torch.matmul(neg_hashtags_tensor, self.weight_h)+torch.matmul(user_tensor, self.weight_h_u)+self.bias_h)

            num_pos = len(pos_hashtag)
            pos_scores = torch.sum(user_specific_video*user_specific_pos_h, dim=1)
            neg_scores = torch.sum(user_specific_video*user_specific_neg_h, dim=1)

            _, index_of_rank_list = torch.topk(torch.cat((neg_scores, pos_scores)), topk)
            index_set = set([iofr.cpu().item() for iofr in index_of_rank_list])
            num_hit = len(index_set.difference(all_set))
            sum_pre += float(num_hit/topk)
            sum_recall += float(num_hit/num_pos)
            idcg = np.sum(1/np.log2(np.arange(min(num_pos, topk))+2))
            ndcg_score = 0.0
            for i in range(num_pos):
                label_pos = neg_num + i
                if label_pos in index_of_rank_list:
                    index = list(index_of_rank_list.cpu().numpy()).index(label_pos)
                    ndcg_score += 1 / math.log(index + 2)

            sum_ndcg += ndcg_score/idcg
        bar.close()

        return sum_pre/sum_item, sum_recall/sum_item
