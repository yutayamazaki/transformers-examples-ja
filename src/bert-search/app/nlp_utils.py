import torch
from transformers import BertJapaneseTokenizer, BertModel


def load_bert_model():
    model_type: str = 'cl-tohoku/bert-base-japanese-whole-word-masking'
    tokenizer = BertJapaneseTokenizer.from_pretrained(model_type)
    model = BertModel.from_pretrained(model_type)
    model.eval()
    return model, tokenizer


def get_sentence_embedding(bert, tokenizer, sentence: str) -> torch.Tensor:
    input_ids = tokenizer.encode(
        sentence, return_tensors='pt'
    )
    result = bert(input_ids)

    # avg_pooled_embed: torch.Tensor = result[0][0].mean(dim=0)
    max_pooled_embed: torch.Tensor = result[0][0].max(dim=0)[0]
    return max_pooled_embed
    # concat_pooled_embed: torch.Tensor = torch.cat(
    #     [bert_pooled_embed, avg_pooled_embed, max_pooled_embed]
    # )
    # return concat_pooled_embed
