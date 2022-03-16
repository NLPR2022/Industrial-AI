from src.data.data import *
from src.data.category_manager import *
from src.data.tokenizer import *
from src.config import *
from src.model.cascade_model import *
def main():
    category_manager = CategoryManager.new_category_manager()
    transformers = get_bert_tokenizer(max_len)
    train_data, test_data = get_category_dataloader(batch_size=batch_size,train_portion=train_portion,filename=train_txt_file)
    model = CascadeModel()
    for epoch in range(num_epochs):
        model.train(train_data)

if __name__ == '__main__':
    main()