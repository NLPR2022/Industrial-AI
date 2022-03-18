import os

import torch
import torch.nn as nn
from tqdm import tqdm

from src.preprocess.data_manager import rawdata_to_sentence

# 임시의 모델입니다...

class LinearClassifier(nn.Module):
    def __init__(self, output_size, hidden_size=768, dropout_rate=None):
        super(LinearClassifier, self).__init__()
        self.classifier = nn.Linear(hidden_size, output_size)
        if dropout_rate:
            self.dropout = nn.Dropout(p=dropout_rate)


class CascadeModel(nn.Module):
    def __init__(self, bert, big_cate_num, mid_cate_num, small_cate_num, transform, device, dr_rate=None):
        super(CascadeModel, self).__init__()
        self.bert = bert.to(device)
        self.device = device
        self.dr_rate = dr_rate
        self.transform = transform
        hidden_size = 768
        self.classifier = nn.Linear(hidden_size, small_cate_num)# 일단은 이것만 해보겠음
        self.big_classifier = LinearClassifier(big_cate_num, dropout_rate=dr_rate)
        self.mid_classifier = nn.ModuleList([LinearClassifier(mid_cate_num, dropout_rate=dr_rate) for _ in range(big_cate_num)])
        self.small_classifier = nn.ModuleList([LinearClassifier(small_cate_num, dropout_rate=dr_rate) for _ in range(mid_cate_num)])

        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(),
                              attention_mask=attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

    def train_model(self, optimizer, train_dataloader, test_dataloader, num_epochs, loss_fn, max_grad_norm, scheduler, log_interval, history):
        best_test_acc = 0
        print(self.inference(rawdata_to_sentence("id_0000006||||철|절삭.용접|카프라배관자재"), self.device))

        for e in range(num_epochs):
            '''
                Train
            '''
            true_positive_count = 0
            all_count = 0
            self.train()
            for batch_id, ((token_ids, valid_length, segment_ids), label) in enumerate(tqdm(train_dataloader)):
                optimizer.zero_grad()
                token_ids = token_ids.long().to(self.device)
                segment_ids = segment_ids.long().to(self.device)
                valid_length = valid_length.to(self.device)
                label = label.to(self.device)
                out = self(token_ids, valid_length, segment_ids)
                loss = loss_fn(out, label)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                true_positive, all_data = self.count_true_positive(out, label)
                true_positive_count += true_positive
                all_count += all_data
                if batch_id % log_interval == 0:
                    train_acc = true_positive_count / all_count
                    print(f"epoch: {e}, batch id: {batch_id}, loss: {loss.data.cpu().numpy()}, train acc: {train_acc}")
            train_acc = true_positive_count / all_count

            '''
                Validation
            '''
            true_positive_count = 0
            all_count = 0
            self.eval()
            for batch_id, ((token_ids, valid_length, segment_ids), label) in enumerate(tqdm(test_dataloader)):
                token_ids = token_ids.long().to(self.device)
                segment_ids = segment_ids.long().to(self.device)
                valid_length = valid_length
                label = label.to(self.device)
                out = self(token_ids, valid_length, segment_ids)
                true_positive, all_data = self.count_true_positive(out, label)
                true_positive_count += true_positive
                all_count += all_data
                if batch_id % log_interval == 0:
                    test_acc = true_positive_count / all_count
                    print(f"epoch: {e}, batch id: {batch_id}, test acc: {test_acc}")
            self.save_model(e, train_acc, test_acc)
            test_acc = true_positive_count / all_count
            if best_test_acc < test_acc:
                best_test_acc = test_acc
                self.save_model(e, train_acc, test_acc)
            history.add_history("epoch", e)
            history.add_history("train_acc", train_acc)
            history.add_history("test_acc", test_acc)
            print(f"epoch: {e}, train acc: {train_acc}, test acc: {test_acc}")

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
        torch.save(self.state_dict(), 'checkpoint/model_epoch_{:.3f}_train_acc_{:.3f}_test_acc_{:.3f}.pth'.format(epoch, train_acc, test_acc))

    def inference(self, sentence, device):
        self.eval()
        (token_ids, valid_length, segment_ids) = self.transform([sentence])
        token_ids = torch.unsqueeze(torch.from_numpy(token_ids),0).to(device)
        valid_length = torch.unsqueeze(torch.from_numpy(valid_length),0).to(device)
        segment_ids = torch.unsqueeze(torch.from_numpy(segment_ids),0).to(device)
        out = self(token_ids, valid_length, segment_ids)
        return torch.argmax(out)

    def inference_by_dataloader(self, test_dataloader):
        out_list = []
        for batch_id, ((token_ids, valid_length, segment_ids), label) in enumerate(tqdm(test_dataloader)):
            token_ids = token_ids.long().to(self.device)
            segment_ids = segment_ids.long().to(self.device)
            valid_length = valid_length
            out = self(token_ids, valid_length, segment_ids)
            for o in out:
                out_list.append((int)(torch.argmax(o).data))
        return out_list

def load_model(model_file):
    return torch.load(model_file)

# def wron