import pandas as pd
from datetime import datetime
import logging
import os
from tqdm_loggable.auto import tqdm
import string
from sklearn.feature_extraction import _stop_words
import numpy as np
import torch
from rank_bm25 import BM25Okapi, BM25L, BM25Plus
from sentence_transformers import CrossEncoder

QUERIES_PATH = os.environ.get(
    "QUERIES_PATH",
    "data/generated_queries/synthesis/",
)

DATA_OUT_DIR = os.environ.get(
    "DATA_OUT_PATH",
    "pfs/out/",
)

NUMBER_DOCUMENTS = os.environ.get(
    "NUMBER_DOCUMENTS",
    "100",  # was 32, file contains 42469 pairs
    # NOTE: number of pairs to consider, set to -1 for full dataset
)

# BM25 ENV VARS

K = os.environ.get(
    "K",
    "1",
    # NOTE: only return most important document
    # changing this changes what goes in JSONL file too
)

# CROSSENCODER ENV VARS

CROSS_ENCODER = os.environ.get(
    "CROSS_ENCODER",
    "cross-encoder/ms-marco-MiniLM-L-12-v2",
)

BATCH_SIZE = os.environ.get(
    "BATCH_SIZE",
    "16",
)

TOP_K = os.environ.get(
    "TOP_K_CROSSENCODER",
    "10000",
    # NOTE: top most similar pairs returned by cross-encoder
)

# ==================================================================================
# BM25 HELPER FUNCTIONS


def bm25_tokenizer(text):
    """Tokenizes a document by converting it to lowercase and removing stopwords.

    Args:
        text (str): Input document

    Returns:
        List[str]: tokenized document
    """
    tokenized_doc = []
    for token in text.lower().split():
        token = token.strip(string.punctuation)  # remove punctuation

        if len(token) > 0 and token not in _stop_words.ENGLISH_STOP_WORDS:
            # append lowercase token if not stopword or empty
            tokenized_doc.append(token)
    return tokenized_doc


def tokenize_corpus(passages, bm25_algorithm):
    """Creates a bm25 instance for a list of texts.

    Args:
        passages (List[str]): List of texts to tokenize
        bm25_algorithm (str): name of bm25-based algorithm to use

    Returns:
        bm25: bm25 instance of indexed documents to be compared to queries
    """
    tokenized_corpus = []
    for passage in tqdm(passages):
        # create list of tokenized texts
        tokenized_corpus.append(bm25_tokenizer(passage))
    # create bm25 instance which indexes documents to be compared to a query
    # https://pypi.org/project/rank-bm25/
    if bm25_algorithm == "bm25":
        bm25 = BM25Okapi(tokenized_corpus)
    elif bm25_algorithm == "bm25l":
        bm25 = BM25L(tokenized_corpus)
    elif bm25_algorithm == "bm25plus":
        bm25 = BM25Plus(tokenized_corpus)
    else:
        print("Invalid BM25 Algorithm")
        exit()

    return bm25


def bm25_search(query_id, bm25, queries, K):
    """Tests to see if the ith document is within the top K most relevant
    documents for the ith query.
    Returns its score if it is, and nothing if it is not.

    Args:
        query_id (int): index of query to compare to texts
        bm25 (bm25): bm25 of indexed texts from tokenize_corpus
        queries (List[str]): list of queries
        K (int): Number of top most relevant documents to consider

    Returns:
        float: bm25 score of document-query pair if in top K most relevant,
        otherwise none
    """
    query = queries[query_id]
    # get bm25 relevance score of each document in the bm25 instance for this one query
    # https://pypi.org/project/rank-bm25/
    bm25_scores = bm25.get_scores(bm25_tokenizer(query))
    # get top K most relevant documents
    top_n = np.argpartition(bm25_scores, -K)[-K:]
    # if document in top K most important, return its score, otherwise none
    if query_id in top_n:
        return bm25_scores[query_id]
    else:
        return None


def main_functionality_bm25_based(queries_file, bm25_algorithm, logger):
    """Main functionality of consistency checks for bm25-based algorithms.
    Generates a JSONL file of good document-query pairs.

    Args:
        queries_file (str): JSONL file containing text and generated queries
        bm25_algorithm (str): name of bm25-based algorithm to use
        logger (logger): logger for command line outputs
    """

    #  get contents of JSONL file as a dataframe
    queries_filepath = os.path.join(QUERIES_PATH, queries_file)
    df = pd.read_json(queries_filepath, lines=True)
    # use first x texts in document
    if NUMBER_DOCUMENTS != -1:
        # optional filtering to only test with a small subset of documents
        # NOTE: otherwise uses full file
        df = df.iloc[:NUMBER_DOCUMENTS]
    # None handling for new query generation prompts
    mask = df["synthetic_query"].isna()
    df = df[~mask]
    df.index = range(len(df))
    # generate lists of texts and queries
    texts = df.text.tolist()
    queries = df.synthetic_query.to_list()
    # index documents using bm25 instance
    bm25 = tokenize_corpus(texts, bm25_algorithm)
    # logging
    start = datetime.now()
    logger.info("started consistency check at: %s" % start)
    logger.info("Number of documents: %s" % len(df))

    good_queries_ids = []
    bm25_scores = []
    for i, query in tqdm(enumerate(queries)):
        # get relevance score of document to query
        score = bm25_search(i, bm25, queries, K)
        # generate list of 'good' pairs and their scores
        # NOTE: DATASET FILTERING!!
        if score is not None:
            good_queries_ids.append(i)
            bm25_scores.append(score)
    #  logging
    logger.info("generation lasted: %s" % (datetime.now() - start))
    logger.info(f"kept {len(good_queries_ids)} (doc,query) pairs out of {len(df)}")
    # convert output to dataframe
    final_df = df.loc[good_queries_ids].copy()
    final_df["bm_25_score"] = bm25_scores
    # convert dataframe to output JSONL file and save
    data_out_path = (
        DATA_OUT_DIR
        + queries_file.split(".")[0]
        + f"_checked_{bm25_algorithm}"
        + ".jsonl"
    )
    final_df.to_json(data_out_path, lines=True, orient="records")


# ==================================================================================
# CROSSENCODER HELPER FUNCTIONS


def sigmoid(x):
    """Function to normalize a score using the sigmoid function.

    Args:
        x (float): score to be normalized

    Returns:
        float: normalized score
    """
    return 1.0 / (1.0 + np.exp(-x))


def cross_encoder_scorer(queries, texts, cross_encoder, batch_size=16):
    """Function to generate sigmid scores for a list of query-text pairs.
    This version of the function has no reranking or filtering - used for
    combination methods for ease of use.

    Args:
        queries (List[str]): list of queries
        texts (List[str]): list of documents
        cross_encoder (CrossEncoder): crossencoder model
        batch_size (int, optional): batch size for crossencoder. Defaults to 16.

    Returns:
        List[float]: list of crossencoder scores corresponding to input pairs
    """
    # generate tuples of queries and associated texts
    query_passage_pairs = [[queries[i], texts[i]] for i in range(len(queries))]
    # predict scores for each query-text pair
    crossencoder_scores = cross_encoder.predict(
        query_passage_pairs, batch_size=batch_size, show_progress_bar=True
    )
    # normalize scores using sigmoid function
    crossencoder_scores = sigmoid(crossencoder_scores)
    return crossencoder_scores


def cross_encoder_reranker(queries, texts, cross_encoder, top_k=10000, batch_size=16):
    """Function to find K most similar document-query pairs and return their
    indexes and scores.

    Args:
        queries (List[str]): list of queries
        texts (List[str]): list of documents
        cross_encoder (CrossEncoder): crossencoder model
        top_k (int, optional): number of top documents to consider. Defaults to 10000.
        batch_size (int, optional): batch size for crossencoder. Defaults to 16.

    Returns:
        List[int]: list of indexes of document-query pairs that are top k most relevant
        List[float]: list of crossencoder scores corresponding to these pairs
    """
    # generate tuples of queries and associated texts
    query_passage_pairs = [[queries[i], texts[i]] for i in range(len(queries))]
    # predict scores for each query-text pair
    crossencoder_scores = cross_encoder.predict(
        query_passage_pairs, batch_size=batch_size, show_progress_bar=True
    )
    # normalize scores using sigmoid function
    crossencoder_scores = sigmoid(crossencoder_scores)
    # get list of top k scores (list of indexes)
    top_n = np.argpartition(crossencoder_scores, -top_k)[-top_k:]
    # get list of top k scores (numerical scores)
    top_scores = crossencoder_scores[top_n]

    return top_n, top_scores


# =================================================================================
# COMBINED METHODS HELPER FUNCTIONS


def main_functionality_combined_filtering(
    queries_file, bm25_algorithm, scoring_method, logger, device
):
    """Main functionality for consistency checks using a combined crossencoder
    and bm25-based algorithm. Uses BM25 to filter the data and then uses a
    cross-encoder to rerank the remaining pairs.

    Args:
        queries_file (str): JSONL file containing text and generated queries
        bm25_algorithm (str): name of bm25-based algorithm to use
        scoring_method (str): linear or crossencoder, type of scoring function to use
        logger (logger): logger for command line outputs
        device (str): cpu or cuda

    """
    #  get contents of JSONL file as a dataframe
    queries_filepath = os.path.join(QUERIES_PATH, queries_file)
    df = pd.read_json(queries_filepath, lines=True)
    # use first x texts in document
    if NUMBER_DOCUMENTS != -1:
        df = df.iloc[:NUMBER_DOCUMENTS]
    # None handling for new query generation prompts
    mask = df["synthetic_query"].isna()
    df = df[~mask]
    df.index = range(len(df))
    # generate lists of texts and queries
    texts = df.text.tolist()
    queries = df.synthetic_query.to_list()
    # index documents using bm25 instance
    bm25 = tokenize_corpus(texts, bm25_algorithm)
    # initialise crossencoder instance
    cross_encoder = CrossEncoder(CROSS_ENCODER.replace("_", "/"), device=device)
    # logging
    start = datetime.now()
    logger.info("started consistency check at: %s" % start)
    logger.info("Number of documents: %s" % len(df))

    # get number of top documents to look at - either k or length of file
    top_k = min(TOP_K, len(texts))

    good_queries_ids = []
    bm25_scores = []
    for i, query in tqdm(enumerate(queries)):
        # get relevance score of document to query
        score = bm25_search(i, bm25, queries, K)
        # generate list of 'good' pairs and their scores - DATASET FILTERING!!
        if score is not None:
            good_queries_ids.append(i)
            bm25_scores.append(score)

    #  generate list of filtered queries to pass to crossencoder
    queries_to_crossencode = [
        queries[i] for i in range(len(texts)) if i in good_queries_ids
    ]
    logger.info(f"kept {len(good_queries_ids)} (doc,query) pairs out of {len(df)}")

    if scoring_method == "linear":
        # get crossencoder scores for document-query pairs
        # NOTE: scores between 0 and 1, not the case for bm25
        cross_scores = cross_encoder_scorer(
            queries_to_crossencode, texts, cross_encoder, batch_size=BATCH_SIZE
        )
        # normalize bm25 scores
        bm25_scores = [
            (a - min(bm25_scores)) / (max(bm25_scores) - min(bm25_scores))
            for a in bm25_scores
        ]
        # Combine the BM25 scores and crossencoder similarities to get the final scores
        final_scores = np.array(
            [
                0.7 * bm25_scores[i] + 0.3 * cross_scores[i]
                for i in range(len(queries_to_crossencode))
            ]
        )
        # get top k most relevant doc-query pairs according to new score
        top_k = min(TOP_K, len(queries_to_crossencode))
        top_n = np.argpartition(final_scores, -top_k)[-top_k:]
        top_scores = final_scores[top_n]
        #  logging
        logger.info("generation lasted: %s" % (datetime.now() - start))
        # convert output to dataframe
        final_df = df.loc[top_n].copy()
        final_df["score"] = top_scores
    elif scoring_method == "crossencoder":
        # get indexes and crossencoder scores for top k most similar
        # document-query pairs
        # NOTE: scores between 0 and 1, not the case for bm25, fixed by sigmoid
        top_k = min(TOP_K, len(queries_to_crossencode))
        cross_good_queries_ids, cross_scores = cross_encoder_reranker(
            queries_to_crossencode,
            texts,
            cross_encoder,
            top_k=top_k,
            batch_size=BATCH_SIZE,
        )
        # simply use crossencoder scores
        final_scores = cross_scores
        #  logging
        logger.info("generation lasted: %s" % (datetime.now() - start))
        # convert output to dataframe
        final_df = df.loc[cross_good_queries_ids].copy()
        final_df["score"] = final_scores
    else:
        print("Invalid scoring method for combined filtering approach!")
        exit()
    # convert dataframe to output JSONL file and save
    data_out_path = (
        DATA_OUT_DIR
        + queries_file.split(".")[0]
        + f"_checked_combined_filter_{bm25_algorithm}_{scoring_method}"
        + ".jsonl"
    )
    final_df.to_json(data_out_path, lines=True, orient="records")


def main_functionality_combined_search(
    queries_file, bm25_algorithm, logger, device, num_queries=int(NUMBER_DOCUMENTS)
):
    """Main functionality for combined consistency check method which uses both a
    bm25-based algorithm and a cross-encoder in the search step for each query.
    This method is slow as the cross-encoder is run on the top 50 bm25-decided
    doc-q pairs for each query q.

    Args:
        queries_file (str): JSONL file containing text and generated queries
        bm25_algorithm (str): name of bm25-based algorithm to use
        logger (logger): logger for command line outputs
        device (str): cpu or cuda
        num_queries (int): number of queries to run the whole process on. Defaults to
        NUMBER_DOCUMENTS.
    """
    # get contents of JSONL file as a dataframe
    queries_filepath = os.path.join(QUERIES_PATH, queries_file)
    df = pd.read_json(queries_filepath, lines=True)
    # None handling for new query generation prompts
    mask = df["synthetic_query"].isna()
    df = df[~mask]
    df.index = range(len(df))
    # generate lists of texts and queries - NOTE: need full lists as want entire corpus
    texts = df.text.tolist()
    queries = df.synthetic_query.to_list()[:num_queries]
    # index documents using bm25 instance
    bm25 = tokenize_corpus(texts, bm25_algorithm)
    cross_encoder = CrossEncoder(CROSS_ENCODER.replace("_", "/"), device=device)
    # logging
    start = datetime.now()
    logger.info("started consistency check at: %s" % start)
    logger.info("Number of documents: %s" % len(df))
    logger.info("Number of queries: %s" % len(queries))

    valids, scores = [], []
    for i in range(len(queries)):
        print(f"Query {i}:")
        # get query to test, and document it is associated with by previous pipeline
        # steps
        query, paired_doc = queries[i], texts[i]
        # generate BM25 scores for all documents regarding this query
        bm25_scores = bm25.get_scores(bm25_tokenizer(query))
        # sort scores in descending order - only consider top 50 documents
        #  NOTE: 50 was decided empirically, see
        # https://adarga.atlassian.net/wiki/spaces/DSRN/pages/3787259912/Consistency+Check+Experiments
        sorted_ind = (-bm25_scores).argsort()[:50]
        # sort documents according to their scores, only use top num_encode
        docs = [texts[a] for a in sorted_ind]
        # generate tuples of queries and associated texts
        query_passage_pairs = [[query, docs[i]] for i in range(len(docs))]
        # generate cross-encoder scores for top num_docs documents (according to bm25)
        # regarding this query
        crossencoder_scores = cross_encoder.predict(
            query_passage_pairs, batch_size=BATCH_SIZE, show_progress_bar=True
        )
        # normalize scores using sigmoid function
        crossencoder_scores = sigmoid(crossencoder_scores)
        # sort scores in descending order
        top_n = np.flip(np.argsort(crossencoder_scores))
        top_scores = crossencoder_scores[top_n]
        top_score = top_scores[0]
        assert top_score == max(crossencoder_scores)
        # sort documents according to scores
        cross_docs = [docs[a] for a in top_n]

        # get document which scored highest for the query
        top_scoring_doc = cross_docs[0]
        # check whether the top scoring document is the document initially paired with
        # the query
        valid = top_scoring_doc == paired_doc
        if valid:
            # if the initial document is still the top result for the query, it is a
            # valid pair - retrieve the score and index of the doc-query pair for
            # file output
            texts_ind = docs.index(cross_docs[0])
            assert docs[texts_ind] == top_scoring_doc
            doc_ind = texts.index(docs[texts_ind])
            assert texts[doc_ind] == top_scoring_doc
            assert query == queries[doc_ind]

            valids.append(doc_ind)
            scores.append(top_score)
    # build output dataframe
    final_df = df.loc[valids].copy()
    final_df["score"] = scores
    logger.info("generation lasted: %s" % (datetime.now() - start))
    # convert dataframe to output JSONL file and save
    data_out_path = (
        DATA_OUT_DIR
        + queries_file.split(".")[0]
        + f"_checked_combined_search_{bm25_algorithm}"
        + ".jsonl"
    )
    final_df.to_json(data_out_path, lines=True, orient="records")


# ==================================================================================
# MAIN CONSISTENCY CHECK FUNCTIONS


def consistency_check_bm25(queries_file, logger):
    """Wrapper function for BM25 consistency check.

    Args:
        queries_file (str): JSONL file containing text and generated queries
        logger (logger): logger for command line outputs
    """
    main_functionality_bm25_based(queries_file, "bm25", logger)


def consistency_check_bm25l(queries_file, logger):
    """Wrapper function for BM25L consistency check.

    Args:
        queries_file (str): JSONL file containing text and generated queries
        logger (logger): logger for command line outputs
    """
    main_functionality_bm25_based(queries_file, "bm25l", logger)


def consistency_check_bm25plus(queries_file, logger):
    """Wrapper function for BM25Plus consistency check.

    Args:
        queries_file (str): JSONL file containing text and generated queries
        logger (logger): logger for command line outputs
    """
    main_functionality_bm25_based(queries_file, "bm25plus", logger)


def consistency_check_crossencoder(queries_file, logger, device):
    """Main function for crossencoder consistency check. Generates a JSONL file
    of good document-query pairs.

    Args:
        queries_file (str): JSONL file containing text and generated queries
        logger (logger): logger for command line outputs
        device (str): cpu or cuda
    """

    #  get contents of JSONL file as a dataframe
    queries_filepath = os.path.join(QUERIES_PATH, queries_file)
    df = pd.read_json(queries_filepath, lines=True)
    # use first x texts in document
    if NUMBER_DOCUMENTS != -1:
        df = df.iloc[:NUMBER_DOCUMENTS]
    # None handling for new query generation prompts
    mask = df["synthetic_query"].isna()
    df = df[~mask]
    df.index = range(len(df))
    # generate lists of texts and queries
    texts = df.text.tolist()
    queries = df.synthetic_query.to_list()
    # instantiate crossencoder (model loaded online)
    cross_encoder = CrossEncoder(CROSS_ENCODER.replace("_", "/"), device=device)
    #  logging
    start = datetime.now()
    logger.info("started consistency check - cross-encoder at: %s" % start)
    logger.info("Number of documents: %s" % len(df))
    # get number of top documents to look at - either k or length of file
    top_k = min(TOP_K, len(texts))
    # get indexes and crossencoder scores for top k most similar document-query pairs
    # NOTE: scores between 0 and 1, not the case for bm25
    good_queries_ids, scores = cross_encoder_reranker(
        queries, texts, cross_encoder, top_k=top_k, batch_size=BATCH_SIZE
    )
    #  logging
    logger.info("cross-encoder computation lasted: %s" % (datetime.now() - start))
    # convert output to dataframe
    final_df = df.iloc[good_queries_ids].copy()
    final_df["cross-encoder-score"] = scores
    final_df = final_df.sort_values("cross-encoder-score", ascending=False)
    # convert dataframe to output JSONL file and save
    data_out_path = (
        DATA_OUT_DIR + queries_file.split(".")[0] + "_checked_crossencoder" + ".jsonl"
    )
    final_df.to_json(data_out_path, lines=True, orient="records")


def consistency_check_combined_search_bm25(queries_file, logger, device):
    """Wrapper function for combined BM25-crossencoder search consistency check.

    Args:
        queries_file (str): JSONL file containing text and generated queries
        logger (logger): logger for command line outputs
        device (str): cpu or cuda
    """
    main_functionality_combined_search(queries_file, "bm25", logger, device)


def consistency_check_combined_search_bm25l(queries_file, logger, device):
    """Wrapper function for combined BM25L-crossencoder search consistency check.

    Args:
        queries_file (str): JSONL file containing text and generated queries
        logger (logger): logger for command line outputs
        device (str): cpu or cuda
    """
    main_functionality_combined_search(queries_file, "bm25l", logger, device)


def consistency_check_combined_search_bm25plus(queries_file, logger, device):
    """Wrapper function for combined BM25Plus-crossencoder search consistency check.

    Args:
        queries_file (str): JSONL file containing text and generated queries
        logger (logger): logger for command line outputs
        device (str): cpu or cuda
    """
    main_functionality_combined_search(queries_file, "bm25plus", logger, device)


def consistency_check_combined_filtering_bm25_cross(queries_file, logger, device):
    """Wrapper function for combined BM25-crossencoder filtering consistency check with
    crossencoder scores.

    Args:
        queries_file (str): JSONL file containing text and generated queries
        logger (logger): logger for command line outputs
        device (str): cpu or cuda
    """
    main_functionality_combined_filtering(
        queries_file, "bm25", "crossencoder", logger, device
    )


def consistency_check_combined_filtering_bm25l_cross(queries_file, logger, device):
    """Wrapper function for combined BM25L-crossencoder filtering consistency check with
    crossencoder scores.

    Args:
        queries_file (str): JSONL file containing text and generated queries
        logger (logger): logger for command line outputs
        device (str): cpu or cuda
    """
    main_functionality_combined_filtering(
        queries_file, "bm25l", "crossencoder", logger, device
    )


def consistency_check_combined_filtering_bm25plus_cross(queries_file, logger, device):
    """Wrapper function for combined BM25Plus-crossencoder filtering consistency check
    with crossencoder scores.

    Args:
        queries_file (str): JSONL file containing text and generated queries
        logger (logger): logger for command line outputs
        device (str): cpu or cuda
    """
    main_functionality_combined_filtering(
        queries_file, "bm25plus", "crossencoder", logger, device
    )


def consistency_check_combined_filtering_bm25_linear(queries_file, logger, device):
    """Wrapper function for combined BM25-crossencoder filtering consistency check with
    linear scores.

    Args:
        queries_file (str): JSONL file containing text and generated queries
        logger (logger): logger for command line outputs
        device (str): cpu or cuda
    """
    main_functionality_combined_filtering(
        queries_file, "bm25", "linear", logger, device
    )


def consistency_check_combined_filtering_bm25l_linear(queries_file, logger, device):
    """Wrapper function for combined BM25L-crossencoder filtering consistency check with
    linear scores.

    Args:
        queries_file (str): JSONL file containing text and generated queries
        logger (logger): logger for command line outputs
        device (str): cpu or cuda
    """
    main_functionality_combined_filtering(
        queries_file, "bm25l", "linear", logger, device
    )


def consistency_check_combined_filtering_bm25plus_linear(queries_file, logger, device):
    """Wrapper function for combined BM25Plus-crossencoder filtering consistency check
    with linear scores.

    Args:
        queries_file (str): JSONL file containing text and generated queries
        logger (logger): logger for command line outputs
        device (str): cpu or cuda
    """
    main_functionality_combined_filtering(
        queries_file, "bm25plus", "linear", logger, device
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # variable setup
    NUMBER_DOCUMENTS = int(NUMBER_DOCUMENTS)
    BATCH_SIZE = int(BATCH_SIZE)
    K = int(K)
    TOP_K = int(TOP_K)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for queries_file in os.listdir(QUERIES_PATH):
        logger = logging.getLogger(f"consistency_check_synthesis - {queries_file}")

        # bm25-based algorithms
        consistency_check_bm25(queries_file, logger=logger)
        consistency_check_bm25l(queries_file, logger=logger)
        consistency_check_bm25plus(queries_file, logger=logger)
        # crossencoder
        consistency_check_crossencoder(queries_file, logger=logger, device=device)
        # combined techniques using search - crossencoder used on each individual
        # query to rank all documents and check pair validity (valid if top document
        # after reranking = document paired to query)
        consistency_check_combined_search_bm25(
            queries_file, logger=logger, device=device
        )
        consistency_check_combined_search_bm25l(
            queries_file, logger=logger, device=device
        )
        consistency_check_combined_search_bm25plus(
            queries_file, logger=logger, device=device
        )
        # combined techniques using filtering - crossencoder used to rerank
        # bm25-filtered pairs (filters a list of pairs to return only the most
        # related)
        # using crossencoder scores
        consistency_check_combined_filtering_bm25_cross(
            queries_file, logger=logger, device=device
        )
        consistency_check_combined_filtering_bm25l_cross(
            queries_file, logger=logger, device=device
        )
        consistency_check_combined_filtering_bm25plus_cross(
            queries_file, logger=logger, device=device
        )
        # using a linear combination of scores
        consistency_check_combined_filtering_bm25_linear(
            queries_file, logger=logger, device=device
        )
        consistency_check_combined_filtering_bm25l_linear(
            queries_file, logger=logger, device=device
        )
        consistency_check_combined_filtering_bm25plus_linear(
            queries_file, logger=logger, device=device
        )
