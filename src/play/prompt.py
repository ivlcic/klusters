import os
import logging
from argparse import ArgumentParser
from datetime import datetime, timezone, timedelta

from typing import List, Dict

from openai import OpenAI

from .corpus.__utils import Params, State, load_range
from .e5 import e5_embed, e5_embed_text
from .utils import compare_clusterings, cluster_louvain, cluster_print, cluster_create_wb, cluster_print_sheet
from .. import CommonArguments
from ..esdl import Elastika
from ..esdl.article import Article
from ..oai.constants import MODEL_TOKENS
from ..oai.embed import openai_embed

logger = logging.getLogger('play.prompt')

cmap = {
    'ccfe00b9-d397-4e85-8310-1a2278ecb73f': 'PS',
    'a65c7372-9fbe-410c-93d7-4613d26488e7': 'DZ',
    '9fb98b28-6e82-4e30-8d36-7e3e9e09a9c0': 'NB',
    '7fd935a6-a1f5-42d1-8b5f-048dd54c07d1': 'NG',
    '011afa08-1b10-48d4-b0ea-cc05d8f7e2a9': 'CD'
}


def add_args(module_name: str, parser: ArgumentParser) -> None:
    # CommonArguments.result_dir(module_name, parser, ('-o', '--result_dir'))
    CommonArguments.tmp_dir(module_name, parser, ('-t', '--tmp_dir'))
    beginning_of_day = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
    parser.add_argument(
        '-s', '--start_date', help='Articles start selection date.', type=str,
        default=beginning_of_day.astimezone(timezone.utc).isoformat()
    )
    next_day = beginning_of_day + timedelta(days=1)
    parser.add_argument(
        '-e', '--end_date', help='Articles end selection date.', type=str,
        default=next_day.astimezone(timezone.utc).isoformat()
    )
    parser.add_argument(
        '-c', '--country', help='Articles selection country.', type=str
    )
    parser.add_argument(
        '-f', '--fields', help='Fields to embed.', type=str, default='tb', required=False,
        choices=['b', 'tb']
    )
    parser.add_argument(
        '-u', '--customer', help='Articles selection customer.', type=str,
        default='a65c7372-9fbe-410c-93d7-4613d26488e7'
    )
    parser.add_argument(
        '-l', '--e5_large', help='Enable large E5.', action='store_true', default=False
    )


def _get_articles(arg):
    requests = Elastika()
    requests.limit(1000)
    requests.filter_customer(arg.customer)
    if arg.country is not None:
        requests.filter_country(arg.country)
    # requests.field('vector_768___textonic_v1')

    articles: List[Article] = requests.gets(arg.start_date, arg.end_date)
    return articles


def cluster_test(arg) -> int:
    text = '''
    Vremenska napoved. 
    Popoldne bo deloma sončno, nastale bodo krajevne plohe in nevihte. V nedeljo bo  pretežno jasno.
    '''
    embeddings = e5_embed_text(arg.tmp_dir, text)
    print(embeddings)
    return 0


def prompt_seba(arg) -> int:
    a_dir = os.path.join(arg.tmp_dir, 'cluster_articles')
    if not os.path.exists(a_dir):
        os.makedirs(a_dir)

    customers = []
    if arg.customer:
        if ',' in arg.customer:
            customers = [part.strip() for part in arg.customer.split(',')]
        else:
            customers = [arg.customer]

    articles: List[Article] = []
    params = Params(arg.start_date, arg.end_date, customers, arg.tmp_dir)

    def callback(s: State, saved_article: Dict, a: Article) -> int:
        articles.append(a)
        return 1

    state = load_range(params, callback)
    logger.info(
        "Loaded [%s] articles for [%s from %s::%s]",
        len(articles), arg.customer, arg.start_date, arg.end_date
    )
    e5_embed(articles, 'e5', arg.tmp_dir, arg.fields, arg.e5_large)

    if arg.customer in cmap.keys():
        arg.customer = cmap[arg.customer]

    append = ''
    if arg.e5_large:
        append = '_large'
    f_prefix = arg.customer + append + '_' + arg.fields + '_' + arg.start_date + '_' + arg.end_date

    print('==========================   E5   ========================== ')
    threshold = 0.96
    e5_l_clusters = cluster_louvain(articles, 'e5', threshold)

    logger.info(
        "Computed [%s] clusters [%s from %s::%s] ",
        len(e5_l_clusters), arg.customer, arg.start_date, arg.end_date
    )
    cluster_print(e5_l_clusters, os.path.join(arg.tmp_dir, 'E5-' + f_prefix + '.txt'))
    wb = cluster_create_wb()
    cluster_print_sheet(wb, "Sheet", e5_l_clusters)
    logger.info(
        "Done [%s] clusters [%s from %s::%s] ",
        len(e5_l_clusters), arg.customer, arg.start_date, arg.end_date
    )

    file_name = os.path.join(
        arg.tmp_dir, f'{f_prefix}.xlsx'
    )
    wb.save(file_name)

    system_prompt = '''
    Prepare a daily overview in Slovenian language of the articles that were published in the media. 

    Divide the daily overview in four paragraphs. 

    In the first paragraph write a general overview of the most important stories that were published. Include at least six stories if they are available. The title of first the paragraph should be in bold format with the title "Najpomembenjše vsebine dneva:"

    Please list six distinct stories as follows:

    1. Point one
    2. Point two
    3. Point three
    4. Point four
    5. Point five
    6. Point six

    Make sure each point is clearly numbered and placed on its own line. 

    In the second paragraph point out potential negative content and the source. The title of the second paragraph should be in bold format with the title "Potencialno kritična vsebina:". 

    In the third paragraph analyze the selected articles and extract the most important quotes, ensuring each quote includes the following details:
    Speaker Identification: Clearly state the name and role (or relevance) of the person quoted.
    Key Message: Summarize the main idea or key point conveyed by the speaker.
    Context (optional but preferred): Briefly mention the surrounding context if it adds clarity to the quote.

    The goal is to identify powerful statements that highlight each speaker's core message or perspective on the subject.


    In the fourth paragraph suggest possible reactions to the published content in few bullet points. The title of the third paragraph should be in bold format with the title "Priporočila za nadalnje aktivnosti:"

    At the end count the number of articles present in the prompt.

    '''

    asistant_prompt = '''

    '''

    clusters = True
    user_prompt = ''
    if clusters:
        for k in e5_l_clusters.keys():
            articles: List[Article] = e5_l_clusters[k]
            for x, a in enumerate(articles):
                # user_prompt += a.title + '\n'
                # user_prompt += a.body + '\n\n\n\n'
                user_prompt += f'Article {x}: \n'
                # user_prompt += f'Article {x}: \n' + a.title + '\n'
                user_prompt += f'Source: ' + a.media + '\n'
                user_prompt += a.body + '\n\n\n\n'
    else:
        for x, a in enumerate(articles):
            user_prompt += f'Article {x}: \n'
            # user_prompt += f'Article {x}: \n' + a.title + '\n'
            user_prompt += f'Source: ' + a.media + '\n'
            user_prompt += a.body + '\n\n\n\n'

    messages = [
        {"role": "system", "content": system_prompt}
    ]
    if asistant_prompt:
        messages.append({"role": "assistant", "content": asistant_prompt})
    messages.append({"role": "user", "content": user_prompt})

    client = OpenAI()
    response = client.chat.completions.create(
        # model='gpt-4o-mini',
        model='gpt-4o',
        seed=2611,
        temperature=1,
        top_p=1,
        messages=messages
    )
    resp: str = response.choices[0].message.content
    resp_file = open(os.path.join(arg.tmp_dir, 'Prompt-' + f_prefix + '.md'), encoding='utf-8', mode='w')
    resp_file.write(resp)
    resp_file.close()
    print(resp)
    return 0
