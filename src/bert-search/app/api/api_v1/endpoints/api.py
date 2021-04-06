from typing import Any, Dict, List, Union

import elasticsearch as es
import numpy as np
import torch
from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import app.nlp_utils as nlp_utils

router = APIRouter()
bert, tokenizer = None, None
INDEX: str = 'documents'


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    sim: torch.Tensor = torch.matmul(a, b) / (a.norm() * b.norm())
    return sim.item()


@router.get('/', response_class=HTMLResponse)
async def index_get(request: Request):
    templates = Jinja2Templates(directory='app/templates')
    return templates.TemplateResponse(
        'index.html',
        context={'request': request, 'results': None, 'query': ''}
    )


def _get_query(query_vector: List[float]):
    return {
        'script_score': {
            'query': {'match_all': {}},
            'script': {
                'source': 'cosineSimilarity(params.query_vector, "vector") + 1.0',
                'params': {'query_vector': query_vector}
            }
        }
    }


@router.post('/', response_class=HTMLResponse)
async def index_post(request: Request, q: str = Form(...)):
    templates = Jinja2Templates(directory='app/templates')

    global bert, tokenizer
    if bert is None or tokenizer is None:
        bert, tokenizer = nlp_utils.load_bert_model()
        bert.eval()
    query_vec: torch.Tensor = nlp_utils.get_sentence_embedding(
        bert, tokenizer, q
    ).detach().cpu().numpy()

    client = es.Elasticsearch('http://elasticsearch:9200')
    resp = client.search(
        index=INDEX,
        body={
            'size': 5,
            'query': _get_query(query_vec.tolist()),
            '_source': {'includes': ['text']}
        }
    )
    client.close()
    res = [{'score': hit['_score'] / 2., 'text': hit['_source']['text']} for hit in resp['hits']['hits']]

    return templates.TemplateResponse(
        'index.html',
        context={'request': request, 'results': res, 'query': q}
    )
