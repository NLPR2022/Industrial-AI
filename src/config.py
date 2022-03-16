import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

train_txt_file = 'data/1. 실습용자료.txt'
category_file = 'data/한국표준산업분류(10차)_국문.xlsx'
category_manager_file = 'etc/category.pkl'
max_len = 64
batch_size = 32
warmup_ratio = 0.1
num_epochs = 4
max_grad_norm = 1
log_interval = 200
learning_rate = 4e-5
train_portion = 0.7