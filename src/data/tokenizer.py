import gluonnlp as nlp
from kobert import get_tokenizer
from kobert import get_pytorch_kobert_model

def get_bert_tokenizer(max_len):
    bertmodel, vocab = get_pytorch_kobert_model()
    tokenizer = get_tokenizer()
    bert_tokenizer = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)
    transform = nlp.data.BERTSentenceTransform(bert_tokenizer, max_seq_length=max_len, pad=True, pair=False)
    return transform