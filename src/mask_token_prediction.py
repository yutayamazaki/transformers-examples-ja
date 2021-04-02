import torch
from transformers import BertForMaskedLM
from transformers import BertJapaneseTokenizer

if __name__ == '__main__':
    model_type: str = 'cl-tohoku/bert-base-japanese-whole-word-masking'
    tokenizer = BertJapaneseTokenizer.from_pretrained(model_type)
    model = BertForMaskedLM.from_pretrained(model_type)
    model.eval()

    input_text: str = f'山田さんが{tokenizer.mask_token}を見たのはこれが初めてでした。巨大だった。'
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    masked_index = \
        torch.where(input_ids == tokenizer.mask_token_id)[1].tolist()[0]

    result = model(input_ids)
    result = result[0][:, masked_index].topk(5).indices.tolist()[0]
    for res in result:
        output = input_ids[0].tolist()
        output[masked_index] = res
        out = tokenizer.decode(output)
        print(out)
