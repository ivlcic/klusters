import logging
import os.path
from typing import List, Union

import torch
from FlagEmbedding import BGEM3FlagModel

from ..esdl.article import Article

logger = logging.getLogger('bgem3.embed')


def bgem3_embed_text(tmp_dir: str, text):
    os.environ['HF_HOME'] = tmp_dir  # local tmp dir set cache
    model = BGEM3FlagModel(
        'BAAI/bge-m3', use_fp16=True,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    embeddings = model.encode([text])['dense_vecs']
    return embeddings[0].tolist()


# noinspection DuplicatedCode
def bgem3_embed(articles: List[Article], embed_field_name: str, tmp_dir: str, fields: str = None,
                cache: Union[str, None] = None):
    os.environ['HF_HOME'] = tmp_dir  # local tmp dir set cache
    model = BGEM3FlagModel(
        'BAAI/bge-m3', use_fp16=True,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    for a in articles:
        if cache is not None and a.from_cache(cache):  # read from file
            if embed_field_name in a.data:  # we already did the embedding ($$$$)
                logger.debug('Loaded %s article BGE-M3 embedding from cache.', a)
                continue
        logger.debug('Loading %s article BGE-M3 embedding ...', a)
        text = a.title + ' ' + a.body
        if fields == 'b':
            text = a.body
            if not text or not text.strip():
                text = a.title
        embeddings = model.encode([text])['dense_vecs']
        logger.info('Loaded %s article BGE-M3 embedding.', a)
        a.data[embed_field_name] = embeddings[0].tolist()
        if cache:
            a.to_cache(cache)  # cache article to file
