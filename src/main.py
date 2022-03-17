from src.data.data import *
from src.data.category_manager import *
from src.data.tokenizer import *
from src.config import *
from src.model.cascade_model import *
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

def main():
    category_manager = CategoryManager.new_category_manager(category_file)
    bert, transform = get_bert_tokenizer(max_len)
    train_dataloader, test_dataloader = get_category_dataloader(batch_size, category_manager, transform, train_portion=0.7, shuffle=True, filename=train_txt_file)

    loss_fn = nn.CrossEntropyLoss()
    model = CascadeModel(bert, category_manager.big_category_num, category_manager.mid_category_num, category_manager.small_category_num, transform, device, dr_rate=0.7)
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

    model.train_model(optimizer, train_dataloader, test_dataloader, num_epochs, loss_fn, max_grad_norm, scheduler, log_interval)

    print(model.inference(rawdata_to_sentence("id_0000006||||철|절삭.용접|카프라배관자재"), device))
    # best model은 아님
    
if __name__ == '__main__':
    main()