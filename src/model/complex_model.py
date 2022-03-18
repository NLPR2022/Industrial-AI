import torch
import torch.nn as nn
from src.model.model import SuperModule
from tqdm import tqdm

class LinearClassifier(nn.Module):
    def __init__(self, output_size, hidden_size=768, dropout_rate=None):
        super(LinearClassifier, self).__init__()
        self.classifier = nn.Linear(hidden_size, output_size)
        if dropout_rate:
            self.dropout = nn.Dropout(p=dropout_rate)

class SimpleModel(SuperModule):
    def __init__(self, bert, big_cate_num, mid_cate_num, small_cate_num, transform, device, dr_rate=None):
        super(SimpleModel, self).__init__()
        self.bert = bert.to(device)
        self.device = device
        self.dr_rate = dr_rate
        self.transform = transform
        hidden_size = 768
        self.big_classifier = LinearClassifier(big_cate_num, dropout_rate=dr_rate)
        self.mid_classifier = nn.ModuleList(
            [LinearClassifier(mid_cate_num, dropout_rate=dr_rate) for _ in range(big_cate_num)])
        self.small_classifier = nn.ModuleList(
            [LinearClassifier(small_cate_num, dropout_rate=dr_rate) for _ in range(mid_cate_num)])

        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(),
                              attention_mask=attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

    def train_model(self, optimizer, train_dataloader, test_dataloader, num_epochs, loss_fn, max_grad_norm, scheduler,
                    log_interval, history):
        best_test_acc = 0

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