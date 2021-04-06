from typing import Any, Dict, List

import elasticsearch as es
import elasticsearch.helpers as helpers

import nlp_utils

INDEX: str = 'documents'
MAPPING: Dict[str, Any] = {
    'mappings': {
        'properties': {
            'text': {'type': 'text'},
            'vector': {'type': 'dense_vector', 'dims': 768},
        }
    }
}


if __name__ == '__main__':
    client = es.Elasticsearch('http://elasticsearch:9200')
    if not client.indices.exists(index=INDEX):
        client.indices.create(index=INDEX, body=MAPPING)

    if client.count(index=INDEX)['count'] >= 5:
        print('Documents already exists. Skip building.')
        exit()

    print('Creating embeddings for ElasticSearch.')
    documents: List[Dict[str, Any]] = [
        {
            'text': '決済方法には何が利用出来ますか？',
        },
        {
            'text': '電話での注文は出来ますか？',
        },
        {
            'text': '送料はいくらかかりますか？',
        },
        {
            'text': '注文から商品の到着まで何日かかりますか？',
        },
        {
            'text': '商品の返品・返金は可能ですか？',
        },
    ]
    bert, tokenizer = nlp_utils.load_bert_model()
    for i, doc in enumerate(documents):
        vec = nlp_utils.get_sentence_embedding(bert, tokenizer, doc['text'])
        doc['vector'] = vec.detach().cpu().numpy().tolist()
        client.create(index=INDEX, id=i+1, body=doc)
    client.close()
