from src.preprocess.data_manager import *
from src.preprocess.category_manager import *
from src.config import *
from src.model.cascade_model import *
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

from src.utils.history_manage import ValueHistory
import datetime

isTrain = False
saved_model_path ='./checkpoint/model_epoch_2.000_train_acc_0.928_test_acc_0.925.pth'
now = datetime.datetime.now()
nowDate = now.strftime('%Y-%m-%d')

def main():
    torch.device(device)

    # 필요한 모듈 불러오는 과정
    history = ValueHistory()
    category_manager = CategoryManager.new_category_manager(category_file)
    bert, transform = get_bert_tokenizer(max_len)

    # data 준비
    train_dataloader, valid_dataloader = get_category_dataloader(batch_size=batch_size, category_manager=category_manager, transform=transform, train_portion=train_portion, filename=labeled_txt_file, pkl_tag=pkl_tag[labeled_txt_file], shuffle=True)

    # 모델 선언
    model = CascadeModel(bert, category_manager.big_category_num, category_manager.mid_category_num, category_manager.small_category_num, transform, device, dr_rate=0.7).to(device)

    if isTrain:
        loss_fn = nn.CrossEntropyLoss()
        # optimizer와 schedule 설정
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
        t_total = len(train_dataloader) * num_epochs
        warmup_step = int(t_total * warmup_ratio)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

        model.train_model(optimizer, train_dataloader, valid_dataloader, num_epochs, loss_fn, max_grad_norm, scheduler, log_interval, history)

    else:
        load_state = load_model(saved_model_path)
        model.load_state_dict(load_state)
        inference_list = model.inference_by_dataloader(valid_dataloader)
        write_wrong_data(inference_list)

    history.save_csv_all_history(f'train_history_{nowDate}', 'history')

def write_wrong_data(inference_list):
    sentence_list = read_raw_txt_file('data/1. 실습용자료.txt')
    different_list = []
    for sentence, out in enumerate(zip(sentence_list, inference_list)):
        small_idx = sentence.split['|'][3]
        out_code = category_manager.id_to_code((int)(out))
        if out_code == small_idx:
            different_list.append(sentence.split('\n')[0] + '|' + out_code)
    with open('data/different.txt', 'w+') as lf:
        lf.write('\n'.join(different_list))

if __name__ == '__main__':
    main()

