from flytekit import task, Resources
from flytekit.deck.renderer import TopFrameRenderer

from sentence_transformers import CrossEncoder
import pandas as pd
import numpy as np
from sklearn.feature_extraction import _stop_words
import string
from tqdm_loggable.auto import tqdm
from rank_bm25 import BM25Okapi, BM25L, BM25Plus
from typing_extensions import Annotated

# region cross-encoder-utils


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


# endregion


# region bm25-utils


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


def main_functionality_bm25_based(df, bm25_algorithm, number_documents):
    """Main functionality of consistency checks for bm25-based algorithms.
    Generates a dataframe of good document-query pairs.

    Args:
        df (DataFrame): Pandas dataframe containing text and generated queries
        bm25_algorithm (str): name of bm25-based algorithm to use
        number_documents (int): Number of pairs to consider
    """
    # use first x texts in document
    if number_documents != -1:
        df = df.iloc[:number_documents]
    # None handling for new query generation prompts
    mask = df["synthetic_query"].isna()
    df = df[~mask]
    df.index = range(len(df))
    # generate lists of texts and queries
    texts = df.text.tolist()
    queries = df.synthetic_query.to_list()
    # index documents using bm25 instance
    bm25 = tokenize_corpus(texts, bm25_algorithm)

    good_queries_ids = []
    bm25_scores = []
    for i, query in tqdm(enumerate(queries)):
        # get relevance score of document to query
        # NOTE: set K to be 1 here as we are only interested in whether the
        # top document is returned
        score = bm25_search(i, bm25, queries, 1)
        # generate list of 'good' pairs and their scores
        # NOTE: DATASET FILTERING!!
        if score is not None:
            good_queries_ids.append(i)
            bm25_scores.append(score)

    # convert output to dataframe
    final_df = df.loc[good_queries_ids].copy()
    final_df["bm_25_score"] = bm25_scores
    return final_df


# endregion


# region combined-utils


def main_functionality_combined_search(
    df,
    bm25_algorithm,
    batch_size,
    number_documents,
    cross_encoder_name,
):
    """Main functionality for combined consistency check method which uses both a
    bm25-based algorithm and a cross-encoder in the search step for each query.
    This method is slow as the cross-encoder is run on the top 50 bm25-decided
    doc-q pairs for each query q.

    Args:
        df (DataFrame): Pandas dataframe containing text and generated queries
        bm25_algorithm (str): name of bm25-based algorithm to use
        batch_size (int): Represents the number of batches the cross encoder will use
        for encoding
        number_documents (int): Number of pairs to consider
        cross_encoder_name (str): the path of the pre-trained cross encoder model to use
    """
    # None handling for new query generation prompts
    mask = df["synthetic_query"].isna()
    df = df[~mask]
    df.index = range(len(df))
    # generate full lists of texts and queries
    texts = df.text.tolist()
    queries = df.synthetic_query.to_list()[:number_documents]
    # index documents using bm25 instance
    bm25 = tokenize_corpus(texts, bm25_algorithm)
    # initialise cross-encoder
    cross_encoder = CrossEncoder(cross_encoder_name)

    valids, scores = [], []
    for i in tqdm(range(len(queries))):
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
            query_passage_pairs, batch_size=batch_size, show_progress_bar=False
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
    return final_df


def main_functionality_combined_filtering(
    df,
    bm25_algorithm,
    scoring_method,
    batch_size,
    number_documents,
    cross_encoder_name,
    TOP_K,
):
    """Main functionality for consistency checks using a combined crossencoder
    and bm25-based algorithm. Uses BM25 to filter the data and then uses a
    cross-encoder to rerank the remaining pairs.

    Args:
        df (pd.DataFrame): Dataframe containing text and generated queries
        batch_size (int): Represents the number of batches the cross encoder will use
        for encoding
        number_documents (int): Number of pairs to consider
        scoring_method (str): scoring method for the combined filtering, can be:
        either 'linear' or 'crossencoder'
        bm25_algorithm (str): bm25 algorithm to use, can be:
        either 'bm25', 'bm25l', or 'bm25plus'
        cross_encoder_name (str): the path of the pre-trained cross encoder model to use
        TOP_K (int): number of most similar pairs returned by cross-encoder
    """
    # None handling for new query generation prompts
    mask = df["synthetic_query"].isna()
    df = df[~mask]
    df.index = range(len(df))
    # use first x texts in document
    if number_documents != -1:
        df = df.iloc[:number_documents]
    # generate lists of texts and queries
    texts = df.text.tolist()
    queries = df.synthetic_query.to_list()
    # index documents using bm25 instance
    bm25 = tokenize_corpus(texts, bm25_algorithm)
    # initialise crossencoder instance
    cross_encoder = CrossEncoder(cross_encoder_name)
    # get number of top documents to look at - either k or length of file
    top_k = min(TOP_K, len(texts))

    good_queries_ids = []
    bm25_scores = []
    for i, query in tqdm(enumerate(queries)):
        # get relevance score of document to query
        # NOTE: set K to be 1 here as we are only interested in whether the
        # top document is returned
        score = bm25_search(i, bm25, queries, 1)
        # generate list of 'good' pairs and their scores - DATASET FILTERING!!
        if score is not None:
            good_queries_ids.append(i)
            bm25_scores.append(score)

    # generate list of filtered queries to pass to crossencoder
    queries_to_crossencode = [
        queries[i] for i in range(len(texts)) if i in good_queries_ids
    ]

    if scoring_method == "linear":
        # get crossencoder scores for document-query pairs
        # NOTE: scores between 0 and 1, not the case for bm25
        cross_scores = cross_encoder_scorer(
            queries_to_crossencode, texts, cross_encoder, batch_size=batch_size
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
            batch_size=batch_size,
        )
        # simply use crossencoder scores
        final_scores = cross_scores
        # convert output to dataframe
        final_df = df.loc[cross_good_queries_ids].copy()
        final_df["score"] = final_scores
    else:
        print("Invalid scoring method for combined filtering approach!")
        exit()

    return final_df


# endregion


# region tasks


@task(
    cache=True,
    cache_version="1.0.3",
    interruptible=True,
    disable_deck=False,
    requests=Resources(cpu="2", mem="5Gi"),
)
def consistency_check_crossencoder(
    df: pd.DataFrame,
    batch_size: int = 16,
    number_documents: int = 32,
    cross_encoder_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
    TOP_K: int = 10000,
) -> Annotated[pd.DataFrame, TopFrameRenderer(10)]:
    """Main function for crossencoder consistency check. Generates a dataframe
    of good document-query pairs.

    Args:
        df (DataFrame): Pandas dataframe containing text and generated queries
        batch_size (int): Represents the number of batches the cross encoder will use
        for encoding
        number_documents (int): Number of pairs to consider
        cross_encoder_name (str): the path of the pre-trained cross encoder model to use
        TOP_K (int): number of most similar pairs returned by cross-encoder
    """
    # use first x texts in document
    if number_documents != -1:
        df = df.iloc[:number_documents]
    # None handling for new query generation prompts
    mask = df["synthetic_query"].isna()
    df = df[~mask]
    df.index = range(len(df))
    # generate lists of texts and queries
    texts = df.text.tolist()
    queries = df.synthetic_query.to_list()
    # instantiate crossencoder (model loaded online)
    cross_encoder = CrossEncoder(cross_encoder_name)
    # get number of top documents to look at - either k or length of file
    top_k = min(TOP_K, len(texts))
    # get indexes and crossencoder scores for top k most similar document-query pairs
    # NOTE: scores between 0 and 1, not the case for bm25
    good_queries_ids, scores = cross_encoder_reranker(
        queries=queries,
        texts=texts,
        cross_encoder=cross_encoder,
        top_k=top_k,
        batch_size=batch_size,
    )
    # convert output to dataframe
    final_df = df.iloc[good_queries_ids].copy()
    final_df["cross-encoder-score"] = scores
    final_df = final_df.sort_values("cross-encoder-score", ascending=False)
    return final_df


@task(
    cache=True,
    cache_version="1.0.3",
    interruptible=True,
    disable_deck=False,
    requests=Resources(cpu="1", mem="2Gi"),
)
def consistency_check_bm25_based(
    df: pd.DataFrame,
    number_documents: int = 32,
    bm25_algorithm: str = "bm25",
) -> Annotated[pd.DataFrame, TopFrameRenderer(10)]:
    """Wrapper function for BM25 consistency check.
    Args:
        df (pd.DataFrame): Dataframe containing text and generated queries
        number_documents (int): Number of pairs to consider
        bm25_algorithm (str): bm25 algorithm to use, can be:
        either 'bm25', 'bm25l', or 'bm25plus'
    """
    return main_functionality_bm25_based(
        df=df, number_documents=number_documents, bm25_algorithm=bm25_algorithm
    )


@task(
    cache=True,
    cache_version="1.0.3",
    interruptible=True,
    disable_deck=False,
    requests=Resources(cpu="2", mem="5Gi"),
)
def consistency_check_combined_search(
    df: pd.DataFrame,
    batch_size: int = 16,
    number_documents: int = 32,
    bm25_algorithm: str = "bm25",
    cross_encoder_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
) -> Annotated[pd.DataFrame, TopFrameRenderer(10)]:
    """Wrapper function for combined BM25Plus-crossencoder filtering consistency check
    with linear scores.
    Args:
        df (pd.DataFrame): Dataframe containing text and generated queries
        batch_size (int): Represents the number of batches the cross encoder will use
        for encoding
        number_documents (int): Number of pairs to consider
        bm25_algorithm (str): bm25 algorithm to use, can be:
        either 'bm25', 'bm25l', or 'bm25plus'
        cross_encoder_name (str): the path of the pre-trained cross encoder model to use
    """
    return main_functionality_combined_search(
        df=df,
        batch_size=batch_size,
        number_documents=number_documents,
        bm25_algorithm=bm25_algorithm,
        cross_encoder_name=cross_encoder_name,
    )


@task(
    cache=True,
    cache_version="1.0.3",
    interruptible=True,
    disable_deck=False,
    requests=Resources(cpu="2", mem="5Gi"),
)
def consistency_check_combined_filtering(
    df: pd.DataFrame,
    batch_size: int = 16,
    number_documents: int = 32,
    scoring_method: str = "linear",
    bm25_algorithm: str = "bm25",
    cross_encoder_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
    TOP_K: int = 10000,
) -> Annotated[pd.DataFrame, TopFrameRenderer(10)]:
    """Wrapper function for combined BM25Plus-crossencoder filtering consistency check
    with linear scores.
    Args:
        df (pd.DataFrame): Dataframe containing text and generated queries
        batch_size (int): Represents the number of batches the cross encoder will use
        for encoding
        number_documents (int): Number of pairs to consider
        scoring_method (str): scoring method for the combined filtering, can be:
        either 'linear' or 'crossencoder'
        bm25_algorithm (str): bm25 algorithm to use, can be:
        either 'bm25', 'bm25l', or 'bm25plus'
        cross_encoder_name (str): the path of the pre-trained cross encoder model to use
        TOP_K (int): number of most similar pairs returned by cross-encoder
    """
    return main_functionality_combined_filtering(
        df=df,
        batch_size=batch_size,
        number_documents=number_documents,
        scoring_method=scoring_method,
        bm25_algorithm=bm25_algorithm,
        cross_encoder_name=cross_encoder_name,
        TOP_K=TOP_K,
    )


# endregion
