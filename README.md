# transformers-examples-ja

にほんごの  
ばーとの  
べんきょう

## リンク
- [huggingface/transformers: 🤗Transformers: State-of-the-art Natural Language Processing for Pytorch and TensorFlow 2.0.](https://github.com/huggingface/transformers)
- [cl-tohoku/bert-japanese: BERT models for Japanese text.](https://github.com/cl-tohoku/bert-japanese)
- [himkt/awesome-bert-japanese: 📝 A list of pre-trained BERT models for Japanese with word/subword tokenization + vocabulary construction algorithm information](https://github.com/himkt/awesome-bert-japanese)

## bert-search

BERTの文章埋め込みを用いた類似文書検索アプリ。

```shell
# Run server
docker-compose -f docker/docker-compose.yaml up
# Execute search
curl "http://localhost:8000/api/search" --get --data-urlencode "q=送料はなんぼですか"
```

## sentence_embedding

事前学習済みモデルを利用して文章の埋め込み表現を得る。

`BertModel.forward`はデフォルトで`last_hidden_state`と`pooler_output`を返す。それぞれの次元は`(batch, seq_len, embedding_dim)` と`(batch, embedding_dim)`であり、`pooler_output`は`lasy_hidden_state`の`[CLS]`トークンに一回linerを通しただけのもの。maxやavgでpoolingしたい場合はコード例のように勝手にやる必要がある。


```shell
$ poetry run python src/sentence_embedding.py
result[0].size():  torch.Size([1, 21, 768])
result[1].size():  torch.Size([1, 768])
bert_pooled_embed.size():  torch.Size([768])
avg_pooled_embed.size() torch.Size([768])
max_pooled_embed.size() torch.Size([768])
concat_pooled_embed.size():  torch.Size([1536])
```

## mask_token_prediction

マスクされた単語が何かを予測するタスクでBertの事前学習に利用されるためそのまま利用できる。

```shell
$ poetry run python src/mask_token_prediction.py

Some weights of the model checkpoint at cl-tohoku/bert-base-japanese-whole-word-masking were not used when initializing BertForMaskedLM: ['cls.seq_relationship.weight', 'cls.seq_relationship.bias']
- This IS expected if you are initializing BertForMaskedLM from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertForMaskedLM from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
[CLS] 山田 さん が ゴジラ を 見 た の は これ が 初めて でし た 。 巨大 だっ た 。 [SEP]
[CLS] 山田 さん が 映画 を 見 た の は これ が 初めて でし た 。 巨大 だっ た 。 [SEP]
[CLS] 山田 さん が これ を 見 た の は これ が 初めて でし た 。 巨大 だっ た 。 [SEP]
[CLS] 山田 さん が それ を 見 た の は これ が 初めて でし た 。 巨大 だっ た 。 [SEP]
[CLS] 山田 さん が 富士山 を 見 た の は これ が 初めて でし た 。 巨大 だっ た 。 [SEP]
```
