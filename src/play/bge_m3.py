import logging
import os.path
import time
import torch

from typing import List, Union
from FlagEmbedding import BGEM3FlagModel

from ..esdl.article import Article

logger = logging.getLogger('bgem3.embed')


def bgem3_embed_text(tmp_dir: str, text):
    os.environ['HF_HOME'] = tmp_dir  # local tmp dir set cache
    model = BGEM3FlagModel(
        'BAAI/bge-m3', use_fp16=True,
        cache_dir=tmp_dir,
        devices='cuda' if torch.cuda.is_available() else 'cpu'
    )
    embeddings = model.encode([text])['dense_vecs']
    return embeddings[0].tolist()


# noinspection DuplicatedCode
def bgem3_embed(articles: List[Article], embed_field_name: str, tmp_dir: str, fields: str = None,
                cache: Union[str, None] = None):
    if torch.cuda.is_available():
        for x in range(0, torch.cuda.device_count()):
            logger.info('Using GPU[%s/%s] %s', x, torch.cuda.device_count(), torch.cuda.get_device_name(x))

    os.environ['HF_HOME'] = tmp_dir  # local tmp dir set cache
    model = BGEM3FlagModel(
        'BAAI/bge-m3',
        # query_max_length=8192, passage_max_length=8192, cache_dir=tmp_dir,
        # devices='cuda:0' if torch.cuda.is_available() else 'cpu'
    )
    start = time.perf_counter()
    for a in articles:
        if cache is not None and a.from_cache(cache):  # read from file
            if embed_field_name in a.data:  # we already did the embedding ($$$$)
                logger.debug('Loaded %s article BGE-M3 embedding from cache.', a)
                continue
        logger.debug('Computing %s article BGE-M3 embedding ...', a)
        text = a.title + ' ' + a.body
        if fields == 'b':
            text = a.body
            if not text or not text.strip():
                text = a.title
        embeddings = model.encode([text], batch_size=12, max_length=8192)['dense_vecs']
        logger.info('Computed %s article BGE-M3 embedding.', a)
        a.data[embed_field_name] = embeddings[0].tolist()
        if cache:
            a.to_cache(cache)  # cache article to file
    logger.info(f'Computed BGE-M3 embeddings in [{((time.perf_counter() - start) * 1000):.3f}]ms.')
