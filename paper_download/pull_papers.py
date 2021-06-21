import json
import os
import sys
import pickle
import re
import time
import pandas as pd
import numpy as np
import logging

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from Bio import Entrez
from multiprocessing import Pool


def process_text(df):
    processed_text = []
    for column in ['text']:
        for i, sentence in enumerate(df[column]):
            if i % 5000 == 0:
                print('%s / %s' % (i, len(df[column])))
            # Convert to lower-case
            sentence = sentence.lower()

            # Removing punctuation
            sentence = re.sub(r'[?|!|\'|"|#]', r'', sentence)
            sentence = re.sub(r'[.|,|)|(|\|/]', r' ', sentence)

            # Removing stop words and stemming
            stemmer = PorterStemmer()
            words = [stemmer.stem(word) for word in sentence.split(
            ) if word not in stopwords.words('english')]

            # Removing digits
            words = [word for word in words if not word.isdigit()]

            processed_text.append(' '.join(words))

        df['processed_%s' % column] = processed_text

    return df


def get_paper_info(start_date, end_date):
    try:
        logging.info(
            'PUBMED QUERY: ("%s"[Date - Create] : "%s"[Date - Create]) ' % (start_date, end_date))
        handle = Entrez.esearch(
            db="pubmed", term='("%s"[Date - Create] : "%s"[Date - Create]) ' % (start_date, end_date), retmax=1000000)
        pmid_records = Entrez.read(handle)
        pmid_list = pmid_records['IdList']
        count = len(pmid_list)
        logging.info('Number of publications being pulled: %s' % count)

        pmid_list = ','.join(pmid_records['IdList'])
        post_xml = Entrez.epost("pubmed", id=pmid_list)
        search_results = Entrez.read(post_xml)

        webenv = search_results["WebEnv"]
        query_key = search_results["QueryKey"]

        batch_size = 300
        all_papers = {}
        missing_abstract = []
        missing_pmid = []
        other_error = []
        for start in range(0, count, batch_size):
            logging.info("Going to download record %i - %i / %i" %
                         (start+1, min(count, start+batch_size), count))

            handle = Entrez.efetch("pubmed", retmode="xml", retmax=batch_size,
                                   retstart=start, webenv=webenv, query_key=query_key)
            records = Entrez.read(handle, validate=False)
            records = records['PubmedArticle']

            for record in records:
                try:
                    paper_info = {}
                    pmid = record.get('MedlineCitation', {}).get('PMID')
                    dict_path = record.get(
                        'MedlineCitation', {}).get('Article', {})
                    abstract = dict_path.get(
                        'Abstract', {}).get('AbstractText')
                    title = dict_path.get('ArticleTitle')
                    publishing_date = dict(
                        dict_path.get('Journal', {}).get('JournalIssue', {}).get('PubDate'))
                    journal = dict_path.get('Journal', {}).get('Title')

                    paper_info = {
                        'abstract': abstract,
                        'title': title,
                        'publishing_date': publishing_date,
                        'journal': journal
                    }

                    if not pmid:
                        missing_pmid.append(pmid)
                        continue

                    elif dict_path == {}:
                        other_error.append(pmid)
                        continue

                    elif abstract == '' or not abstract:
                        missing_abstract.append(pmid)
                        continue

                    all_papers[pmid] = paper_info

                except Exception as e:
                    logging.debug(e)
                    other_error.append(pmid)
                    continue

        logging.info('Completed pulling papers')
        logging.info('# of papers with empty abstracts: %s' %
                     len(missing_abstract))
        logging.info('# of PMIDs that could not be pulled: %s' %
                     len(missing_pmid))
        logging.info('# of papers with other errors: %s' % len(other_error))
        logging.info('Total # of papers ignored: %s' % (
            len(missing_abstract)+len(missing_pmid)+len(other_error)))
        errors = dict(missing_abstract=missing_abstract,
                      missing_pmid=missing_pmid, other_error=other_error)
        return all_papers, errors, count
    except Exception as e:
        logging.error(e)
        raise e


def get_pubmed_df(data):
    text = [' '.join(data[pmid]['abstract'])
            for pmid in data if data[pmid] != False]
    journals = [data[pmid]['journal'] for pmid in data if data[pmid] != False]
    titles = [data[pmid]['title'] for pmid in data if data[pmid] != False]
    pmids = [pmid for pmid in data if data[pmid] != False]

    df = pd.DataFrame()
    df['text'] = text
    df['title'] = titles
    df['journal'] = journals
    df['pmid'] = pmids

    return df


def main():
    logging.root.setLevel(logging.NOTSET)
    Entrez.email = ''
    Entrez.api_key = ''
    CONFIG_PATH = 'jun_2017_to_dec_2020_config.json'
    with open(CONFIG_PATH, 'r') as file:
        dates = json.load(file)

    def parallelize_dataframe(df, func, n_cores=4):
        df_split = np.array_split(df, n_cores)
        pool = Pool(n_cores)
        df = pd.concat(pool.map(func, df_split))
        pool.close()
        pool.join()
        return df

    for start_date, end_date in dates:
        all_papers, *_ = get_paper_info(start_date, end_date)
        df = get_pubmed_df(all_papers)
        pubmed_papers_df = parallelize_dataframe(df, process_text)

        file_name = '%s_to_%s.json' % (start_date, end_date)
        file_name = file_name.replace('/', '-')
        PATH = os.path.join('out', 'jun-2017_to_dec-2020', file_name)
        pubmed_papers_df.to_json(PATH)


if __name__ == '__main__':
    main()
