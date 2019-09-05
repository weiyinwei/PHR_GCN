import random
import time
from tqdm import tqdm
import numpy as np
from collections import defaultdict

path = '/Hashtag_recom/Data/'
video_file = path+'video_list.txt'
user_file = path+'user_list.txt'
hashtag_file = path+'hashtag_list.txt'
video_info_file = path+'video_info_offset.txt'

video_list = open(video_file).readlines()
user_list = open(user_file).readlines()
hashtag_list = open(hashtag_file).readlines()

num_video = len(video_list)
num_user = len(user_list)
num_hashtag = len(hashtag_list)

user_video_dict = defaultdict(set)
video_hashtag_list = list()

video_info_list = open(video_info_file).readlines()
for index, video_info in enumerate(video_info_list):
	user, hashtags = video_info[:-1].split(';')
	hashtag_list = hashtags.split(':')
	user_video_dict[int(user)].add(num_user+num_hashtag+index)
	video_hashtag_list.append(list(map(int, hashtag_list)))
all_train_list = []
all_val_list = []
all_test_list = []

all_hashtag = set(range(num_user, num_user+num_hashtag))

bar = tqdm(total=len(user_video_dict.keys()))
for key in user_video_dict.keys():
	bar.update(1)
	items = user_video_dict[key]
	num_items = len(items)
	num_train = int(num_items*0.8)
	train_items = random.sample(items, num_train)
	for item in train_items:
		hashtags = video_hashtag_list[item-num_hashtag-num_user]
		all_train_list.extend(list(zip(list(zip([key]*len(hashtags), [item]*len(hashtags))), hashtags)))

	val_test_items = items.difference(train_items)
	num_val = int(len(val_test_items)*0.5)
	val_items = random.sample(val_test_items, num_val)
	for item in val_items:
		hashtags = video_hashtag_list[item-num_hashtag-num_user]
		neg_hashtag = random.sample(all_hashtag.difference(hashtags), 1000)
		all_val_list.append([key, item]+list(neg_hashtag)+hashtags)
	test_items = val_test_items.difference(val_items)
	if len(test_items) > 0:
		for item in test_items:
			hashtags = video_hashtag_list[item-num_hashtag-num_user]
			neg_hashtag = random.sample(all_hashtag.difference(hashtags), 1000)
			all_test_list.append([key, item]+list(neg_hashtag)+hashtags)
bar.close()

np.save(path+'video_hashtag', np.array(video_hashtag_list))
np.save(path+'train', np.array(all_train_list))
np.save(path+'val', np.array(all_val_list))
np.save(path+'test', np.array(all_test_list))

