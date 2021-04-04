import uuid
from typing import Any, Dict, List

import numpy as np
import torch
from fastapi import APIRouter
from transformers import BertJapaneseTokenizer, BertModel

import app.nlp_utils as nlp_utils
from app.nlp_utils import get_sentence_embedding, load_bert_model

router = APIRouter()
bert, tokenizer = None, None

index: List[Dict[str, str]] = [
    {
        'text': '決済方法には何が利用出来ますか？',
        'vector': None
    },
    {
        'text': '電話での注文は出来ますか？',
        'vector': None
    },
    {
        'text': '送料はいくらかかりますか？',
        'vector': None
    },
    {
        'text': '注文から商品の到着まで何日かかりますか？',
        'vector': None
    },
    {
        'text': '商品の返品・返金は可能ですか？',
        'vector': None
    },
]


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    sim: torch.Tensor = torch.matmul(a, b) / (a.norm() * b.norm())
    return sim.item()


@router.get('/search')
async def search(q: str):
    global bert, tokenizer, index
    if bert is None or tokenizer is None:
        bert, tokenizer = nlp_utils.load_bert_model()
        bert.eval()

    query_vec: torch.Tensor = nlp_utils.get_sentence_embedding(
        bert, tokenizer, q
    )
    results = {
        'texts': [],
        'scores': []
    }
    for idx, item in enumerate(index):
        text: str = item['text']
        vector = item['vector']
        if vector is None:
            vector = nlp_utils.get_sentence_embedding(bert, tokenizer, text)
        index[idx] = {'text': item['text'], 'vector': vector}
        results['texts'].append(text)
        results['scores'].append(cosine_similarity(vector, query_vec))

    indices = np.argsort(results['scores'])[::-1]
    resp = []
    for idx in indices:
        resp.append({
            'text': results['texts'][idx],
            'score': results['scores'][idx]
        })
    for r in resp:
        print(r)
    return resp
