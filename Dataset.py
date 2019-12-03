import random
import time
import torch
import numpy as np 
from collections import defaultdict
from torch_geometric.data import Data, Dataset, DataLoader
from tqdm import tqdm

class BaseDataset:
	def __init__(self, path, filename):
		self.user_file = 'user_list.txt'
		self.hashtag_file = 'hashtag_list.txt'
		self.video_file = 'video_list.txt'
		self.path = path
		self.filename = filename
		self.video_hashtag_file = 'video_hashtag.npy'
		self.datas = np.load(self.path+self.filename, allow_pickle=True)
		self.all_hashtag_set = set()
		self.v_h_list = np.load(self.path+self.video_hashtag_file, allow_pickle=True)
		for data in self.datas:
			_, hashtag = data
			self.all_hashtag_set.add(hashtag)
		self.num_user, self.num_hashtag, self.num_video = self.get_num()

	def get_num(self):
		with open(self.path+self.user_file, 'r') as f:
			self.user_list = f.readlines()
			num_user = len(self.user_list)

		with open(self.path+self.hashtag_file, 'r') as f:
			self.hashtag_list = f.readlines()
			num_hashtag = len(self.hashtag_list)			
		
		with open(self.path+self.video_file, 'r') as f:
			self.video_list = f.readlines()
			num_video = len(self.video_list)
		
		return num_user, num_hashtag, num_video

	def get_edges(self):
		self.all_edges_set = set()
		for index, data in enumerate(self.datas):
			u_v, hashtag = data
			user, video = u_v
			self.all_edges_set.add((user, video))
			self.all_edges_set.add((hashtag,video))
			self.all_edges_set.add((user, hashtag))
			self.all_edges_set.add((hashtag, user))
		return self.all_edges_set


	def get_heter_edges(self):
		self.uh_edges_set = set()
		self.v_uh_edges_set = set()
		# self.v_u_edges_set = set()
		# self.v_h_edges_set = set()
		for index, data in enumerate(self.datas):
			u_v, hashtag = data
			user, video = u_v
			# self.v_u_edges_set.add((user, video))
			# self.v_h_edges_set.add((hashtag,video))
			self.v_uh_edges_set.add((video, user))###############test
			self.v_uh_edges_set.add((video, hashtag))###############test
			self.uh_edges_set.add((user, hashtag))
			self.uh_edges_set.add((hashtag, user))
		return self.uh_edges_set, self.v_uh_edges_set#self.v_u_edges_set, self.v_h_edges_set

	def __len__(self):
		return len(self.datas)

	def __getitem__(self, index):
		u_v, pos_hashtag = self.datas[index]
		user, video = u_v
		neg_hashtag = random.sample(self.all_hashtag_set.difference(self.v_h_list[video-self.num_user-self.num_hashtag]), 1)[0]
		
		return torch.tensor([user, video, pos_hashtag, neg_hashtag],dtype=torch.long)#torch.tensor(self.datas[index] ,dtype=torch.long).squeeze()


# class YFCCDataset(Dataset):
# 	def __init__(self, path, num_neigh=30):
# 		super(YFCCDataset, self).__init__(path, None, None)
# 		self.path = path
# 		self.video_hashtag_file = 'video_hashtag.npy'
# 		self.train_file = 'train.npy'
# 		self.user_file = 'user_list.txt'
# 		self.hashtag_file = 'hashtag_list.txt'
# 		self.video_file = 'video_list.txt'
# 		self.train_datas = np.load(self.path+self.train_file)
# 		self.u_h_dict = defaultdict(set)
# 		self.h_u_dict = defaultdict(set)
# 		self.u_v_dict = defaultdict(set)
# 		self.h_v_dict = defaultdict(set)
# 		self.v_h_list = np.load(self.path+self.video_hashtag_file)
# 		self.all_hashtag_set = set()
# 		self.num_user, self.num_hashtag, self.num_video = self.get_num()
# 		self.num_neigh = num_neigh
# 		self.gen_dict()

# 	def get_num(self):
# 		with open(self.path+self.user_file, 'r') as f:
# 			self.user_list = f.readlines()
# 			num_user = len(self.user_list)

# 		with open(self.path+self.hashtag_file, 'r') as f:
# 			self.hashtag_list = f.readlines()
# 			num_hashtag = len(self.hashtag_list)			
		
# 		with open(self.path+self.video_file, 'r') as f:
# 			self.video_list = f.readlines()
# 			num_video = len(self.video_list)
		
# 		return num_user, num_hashtag, num_video

# 	def gen_dict(self):
# 		for index, data in enumerate(self.train_datas):
# 			u_v, hashtag = data
# 			user, video = u_v
# 			self.all_hashtag_set.add(hashtag)
# 			self.u_h_dict[user].add(hashtag)
# 			self.h_u_dict[hashtag].add(user)
# 			self.u_v_dict[user].add(video)
# 			self.h_v_dict[hashtag].add(video)

# 	def _download(self):
# 		pass
# 	def _process(self):
# 		pass
# 	def __len__(self):
# 		return len(self.train_datas)

# 	def __subgraph__(self, node, dict1, dict2, dict3, video_node, bias=0):
# 		# In this part, to represent the user, we construct a subgraph which contains user, his hashtags and the micro-videos.
# 		# We get the edges between user and hashtags and the edges between his hashtags and correpsonding micro-video.
# 		# user_nodes contains user, hashtags, and micro-videos, like [user, hashtags, micro-videos]
# 		# user_edges can be viewed as two parts, one is [[user, hashtag_1], ..., [user, hashtag_M]], and the other is [[hastag1, micro-video1]...[hashtag_M, micro-video_N]]
# 		all_nodes_list = [node]
# 		first_nodes_set = dict1[node]#get hashtags connected with the user
# 		first_nodes_list = list(random.sample(first_nodes_set, self.num_neigh) if len(first_nodes_set)>self.num_neigh else first_nodes_set)
# 		first_edges = list(zip([node]*len(first_nodes_list), first_nodes_list))#yield the user-hashtag edges 
# 		all_nodes_list += first_nodes_list#obtain the nodes of user and corresponding hashtag nodes
# 		second_edges = list()# used to retore the <hashtag, video> edge
# 		all_temp_nodes = list(dict2[node].difference({video_node}))# get all user video edge
# 		for temp in first_nodes_list:#u_h_nodes:
# 			temp_nodes = dict3[temp].difference({video_node})#get the video nodes correponds to the hashtag 
# 			temp_nodes = list(temp_nodes.intersection(all_temp_nodes))# get the video of the user corresponds to hashtag h
# 			all_nodes_list += temp_nodes
# 			second_edges += list(zip([temp]*len(temp_nodes), temp_nodes))# build the egdes between hashtag h and videos for user u
# 		all_edges_list = first_edges+second_edges
# 		num_nodes = len(all_nodes_list)
# 		num_first_edges = len(first_edges)
# 		num_second_edges = len(second_edges)
# 		num_edges = len(all_edges_list)
# 		nodes_tensor = torch.tensor(all_nodes_list, dtype=torch.long)
# 		edges_tensor = torch.tensor(all_edges_list, dtype=torch.long).t()
# 		max_node = torch.max(nodes_tensor)
# 		relabel_tensor = torch.zeros(max_node+1, dtype=torch.long)
# 		relabel_tensor[nodes_tensor] = torch.arange(bias, bias+len(nodes_tensor), dtype=torch.long)
# 		relabel_nodes_tensor = relabel_tensor[nodes_tensor]
# 		relabel_edges_tensor = torch.cat((relabel_tensor[edges_tensor[0]].unsqueeze(0), relabel_tensor[edges_tensor[1]].unsqueeze(0)), dim=0)

# 		return nodes_tensor, relabel_edges_tensor, num_nodes, num_first_edges, num_second_edges, num_edges

# 	def get(self, index):
# 		u_v, pos_hashtag = self.train_datas[index]
# 		user, video = u_v
# 		neg_hashtag = random.sample(self.all_hashtag_set.difference(self.v_h_list[video-self.num_user-self.num_hashtag]), 1)[0]
# 		########################################################################################################################################################################################################
# 		user_nodes, user_edges, num_user_nodes, num_u_h_edges, num_h_v_edges, num_user_edges = self.__subgraph__(user, self.u_h_dict, self.u_v_dict, self.h_v_dict, video)
# 		pos_h_nodes, pos_h_edges, num_pos_h_nodes, num_pos_h_u_edges, num_pos_u_v_edges, num_pos_h_edges = self.__subgraph__(pos_hashtag, self.h_u_dict, self.h_v_dict, self.u_v_dict, video, bias=num_user_nodes)
# 		neg_h_nodes, neg_h_edges, num_neg_h_nodes, num_neg_h_u_edges, num_neg_u_v_edges, num_neg_h_edges = self.__subgraph__(neg_hashtag, self.h_u_dict, self.h_v_dict, self.u_v_dict, video, bias=num_user_nodes+num_pos_h_nodes)
# 		item = Data()
# 		item['edge_index'] = torch.cat((torch.cat((user_edges, pos_h_edges), dim=1), neg_h_edges), dim=1)
# 		item['nodes'] = torch.cat((torch.cat((user_nodes, pos_h_nodes)), neg_h_nodes))
# 		item['num_nodes'] = torch.tensor(num_user_nodes+num_pos_h_nodes+num_neg_h_nodes, dtype=torch.long)
# 		item['num_nodes_'] = torch.tensor([num_user_nodes, num_pos_h_nodes, num_neg_h_nodes], dtype=torch.long)
# 		item['num_first_edges'] = torch.tensor([num_u_h_edges, num_pos_h_u_edges, num_neg_h_u_edges], dtype=torch.long)
# 		item['num_second_edges'] = torch.tensor([num_h_v_edges, num_pos_u_v_edges, num_neg_u_v_edges], dtype=torch.long)
# 		item['videos'] = torch.tensor([video], dtype=torch.long)
# 		item['users'] = torch.tensor([user], dtype=torch.long)
# 		item['pos_hashtags'] = torch.tensor([pos_hashtag], dtype=torch.long)
# 		item['neg_hashtags'] = torch.tensor([neg_hashtag], dtype=torch.long)
# 		return item



