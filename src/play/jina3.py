import logging
import os.path
from typing import List, Union

from transformers import AutoModel

from ..esdl.article import Article

logger = logging.getLogger('bgem3.embed')


def jina3_embed_text(tmp_dir: str, text):
    model = AutoModel.from_pretrained(
        'jinaai/jina-embeddings-v3', trust_remote_code=True,
        device_map='auto', cache_dir=os.path.join(tmp_dir, 'jina3')
    )
    embeddings = model.encode([text])['dense_vecs']
    return embeddings[0].tolist()


# noinspection DuplicatedCode
def jina3_embed(articles: List[Article], embed_field_name: str, tmp_dir: str, fields: str = None,
                cache: Union[str, None] = None):
    os.environ['HF_HOME'] = tmp_dir  # local tmp dir set cache
    model = AutoModel.from_pretrained(
        'jinaai/jina-embeddings-v3', trust_remote_code=True,
        device_map='auto', cache_dir=os.path.join(tmp_dir, 'jina3')
    )

    for a in articles:
        if cache is not None and a.from_cache(cache):  # read from file
            if embed_field_name in a.data:  # we already did the embedding ($$$$)
                logger.debug('Loaded %s article Jina3 embedding from cache.', a)
                continue
        logger.debug('Loading %s article Jina3 embedding ...', a)
        text = a.title + ' ' + a.body
        if fields == 'b':
            text = a.body
            if not text or not text.strip():
                text = a.title
        embeddings = model.encode([text], task='text-matching', show_progress_bar=False)
        logger.info('Loaded %s article Jina3 embedding.', a)
        a.data[embed_field_name] = embeddings[0].tolist()
        if cache:
            a.to_cache(cache)  # cache article to file
