import torch
from transformers import BertJapaneseTokenizer, BertModel

if __name__ == '__main__':
    model_type: str = 'cl-tohoku/bert-base-japanese-whole-word-masking'
    tokenizer = BertJapaneseTokenizer.from_pretrained(model_type)
    model = BertModel.from_pretrained(model_type)
    model.eval()

    sentence: str = '山田さんがゴジラを見たのはこれが初めてでした。巨大だった。'
    # デフォルトでadd_special_tokens=Trueなので勝手に[CLS]と[SPE]をつけてくれる
    input_ids = tokenizer.encode(
        sentence, return_tensors='pt'
    )  # (1, 21) : (batch, seq_len)

    # last_hidden_stateとpooler_outputが帰ってくる
    result = model(input_ids)
    # result[0]がlast_hidden_stateでそれをpoolingしたものがresult[1]のpooler_output
    # pooler_outputは[CLS]トークンを取得しているだけなので、avgやmaxでpoolingした場合は勝手に計算する必要がある
    print(
        'result[0].size(): ', result[0].size()
    )  # (1, 21, 768) : (batch, seq_len, embedding_dim)
    print(
        'result[1].size(): ', result[1].size()
    )  # (1, 768) : (batch, embedding_dim)

    bert_pooled_embed: torch.Tensor = result[1][0]
    avg_pooled_embed: torch.Tensor = result[0][0].mean(dim=0)
    max_pooled_embed: torch.Tensor = result[0][0].max(dim=0)[0]
    concat_pooled_embed: torch.Tensor = torch.cat(
        [avg_pooled_embed, max_pooled_embed]
    )
    print('bert_pooled_embed.size(): ', bert_pooled_embed.size())  # (768, )
    print('avg_pooled_embed.size()', avg_pooled_embed.size())  # (768, )
    print('max_pooled_embed.size()', max_pooled_embed.size())  # (768, )
    print('concat_pooled_embed.size(): ', concat_pooled_embed.size())  # (1536, )
