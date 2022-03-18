import os

import torch
import torch.nn as nn
from tqdm import tqdm

from src.preprocess.data_manager import rawdata_to_sentence


class SuperModule(nn.Module):
    def __init__(self, bert, big_cate_num, mid_cate_num, small_cate_num, transform, device, dr_rate=None):
        super().__init__(bert, big_cate_num, mid_cate_num, small_cate_num, transform, device, dr_rate=None)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        super().forward(token_ids, valid_length, segment_ids)

    @staticmethod
    def count_true_positive(pred, label):
        max_pred_vals, max_pred_indices = torch.max(pred, 1)
        max_label_vals, max_label_indices = torch.max(label, 1)
        count = 0
        for pred_idx, label_idx in zip(max_pred_indices, max_label_indices):
            if pred_idx == label_idx:
                count = count + 1
        return count, max_pred_indices.size()[0]

    def save_model(self, epoch, train_acc, test_acc):
        if os.path.exists("checkpoint") == False:
            os.mkdir("checkpoint")
        torch.save(self.state_dict(),
                   'checkpoint/model_epoch_{:.3f}_train_acc_{:.3f}_test_acc_{:.3f}.pth'.format(epoch, train_acc,
                                                                                               test_acc))

    def inference(self, sentence):
        self.eval()
        (token_ids, valid_length, segment_ids) = self.transform([sentence])
        token_ids = torch.unsqueeze(torch.from_numpy(token_ids), 0).to(self.device)
        valid_length = torch.unsqueeze(torch.from_numpy(valid_length), 0).to(self.device)
        segment_ids = torch.unsqueeze(torch.from_numpy(segment_ids), 0).to(self.device)
        out = self(token_ids, valid_length, segment_ids)
        return torch.argmax(out)

    def inference_by_dataloader(self, test_dataloader):
        out_list = []
        for batch_id, ((token_ids, valid_length, segment_ids), label) in enumerate(tqdm(test_dataloader)):
            token_ids = token_ids.long().to(self.device)
            segment_ids = segment_ids.long().to(self.device)
            valid_length = valid_length
            out = self(token_ids, valid_length, segment_ids)
            out_list.append(torch.argmax(out).data)
        return out_list

def load_model(model_file):
    return torch.load(model_file)

