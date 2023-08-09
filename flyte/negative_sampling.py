import pandas as pd
from tqdm_loggable.auto import tqdm
from rank_bm25 import BM25Okapi, BM25L, BM25Plus
import string
from sklearn.feature_extraction import _stop_words
import numpy as np
from rake_nltk import Rake
import nltk
from typing_extensions import Annotated
from flytekit import task, Resources
from flytekit.deck.renderer import TopFrameRenderer


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
        bm25_algorithm(str): name of BM25-based algorithm to use

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
        print("Invalid BM25 algorithm!")
        exit()

    return bm25


def bm25_search(query, bm25, K):
    """Generates similarity scores for top K most relevant
    documents regarding the input query.

    Args:
        query (str): input query to compare to documents
        bm25 (bm25): bm25 of indexed texts from tokenize_corpus
        K (int): Number of top most relevant documents to consider

    Returns:
        List[int]: indexes of remaining documents
        List[float': bm25 scores of document-query pairs if in top K most relevant
    """
    bm25_scores = bm25.get_scores(bm25_tokenizer(query))
    # filter out top K most relevant
    # NOTE: documents are ORDERED in terms of relevance to the query
    top_n = np.argpartition(bm25_scores, -K)[-K:]

    return top_n, bm25_scores[top_n]


# endregion

# region sampling-utils


def negative_sample_random_choice(top_n, N, texts, scores):
    """Performs random choice negative sampling for a given set of documents.

    Args:
        top_n (List[int]): Sample IDs from the first top_n documents in list of texts
        N (int): Number of documents to return
        texts (List[str]): List of all documents to sample from
        scores (List[float]): List of bm25 scores assigned to documents

    Returns:
        List[(str, float)]: Texts and corresponding bm25 scores for documents chosen by
        the sample
    """
    # generate list of random indexes to use
    samples_idx = np.random.choice(range(len(top_n)), N, replace=False)
    # return text and bm25 score for these indexes
    return [(texts[top_n[x]], scores[x]) for x in samples_idx]


def negative_sample_from_topK(query, bm25, N, texts):
    """Performs negative sampling while only considering the top_k most relevant
    documents to a query according to bm25. Returns a sample of N documents and their
    scores.

    Args:
        query (str): Input query to compare to all texts
        bm25 (bm25): bm25 of indexed texts from tokenize_corpus
        N (int): Number of negative samples to return
        texts (List[str]): List of all documents to compare to query

    Returns:
        List[(str, float)]: Texts and corresponding bm25 scores for documents chosen by
        the sample
    """
    # generate document IDs and bm25 scores for each document regarding one query
    # NOTE: docs are ordered in terms of relevance to the query
    # NOTE: only want to consider 50 most similar documents to query
    # 50 was decided empirically
    docs, scores = bm25_search(query, bm25, 50)
    # generate sample of documents to use as negative samples for the query
    # NOTE: always disregard top 5 documents as we want the negatively sampled
    # document to be similar but not too similar (we assume 2 - 4 will be very similar)
    negatives = negative_sample_random_choice(docs[5:], N, texts, scores[5:])
    # return documents to be used as negative sample for the query
    return negatives


def generate_batches(query, all_queries, bm25, batch_size):
    """Generates a batch of queries for a given query. Used for batch-based
    negative sampling techniques.

    Args:
        query (str): Query to generate batch for
        all_queries (List[str]): List of all queries in corpus
        bm25 (bm25): bm25 of indexed texts from tokenize_corpus
        batch_size (int): Size of batches to generate

    Returns:
        List(str): List of queries in generated batch, including original query
    """
    # generate similarity scores for all queries compared to inout query
    # generate list of queries and scores as output
    queries, scores = bm25_search(query, bm25, len(all_queries))
    out = [(all_queries[queries[i]], scores[i]) for i in range(len(queries) - 1)]
    # remove duplicates and change to descending order
    out = list(set(out))
    out = sorted(out, key=lambda x: x[1], reverse=True)[: batch_size * 2]
    # we dont want most similar samples, want to be similar but not too similar
    closest = [a for (a, b) in out]
    closest = closest[5 : batch_size + 4]
    # add original query to its batch
    closest.insert(0, query)
    assert len(closest) == batch_size
    return closest


# endregion

# region main-functionality


def negative_sampling_bm25(
    checked_df,
    full_df,
    bm25_algorithm,
    sample_technique,
    batch_size,
    negative_samples,
    number_documents,
):
    """Main functionality for negative sampling. Reads a file of document-query pairs
    and generates N negative sample documents for each query.

    Args:
        checked_df (DataFrame): Pandas dataframe output from consistency check
        full_df (DataFrame): Dataframe containing all queries and docs in corpus
        bm25_algorithm (str): bm25 algorithm to use, can be:
        either 'bm25', 'bm25l', or 'bm25plus'
        sample_technique (str): sampling technique to use, can be:
        either 'random', 'gold_batch', or 'keywords'
        batch_size (int): batch size for batch-based negative sampling techniques
        negative_samples (int): number of documents to choose as negative
        samples for a query
        number_documents (int): number of pairs to consider
    """
    # format input dataframe
    # NOTE: there should be an input file for each consistency check technique
    checked_df.index = range(len(checked_df))

    # format dataframe containing full sample of texts and queries
    full_df.index = range(len(full_df))

    if number_documents != -1:
        # optional filtering to only test with a small subset of documents
        # NOTE: otherwise uses full file
        checked_df = checked_df.iloc[:number_documents]

    # set up variables - filtered list of queries, full corpus of documents and queries
    sample_texts = full_df.text.tolist()
    sample_queries = full_df.synthetic_query.tolist()
    # filtered version
    paired_docs = checked_df.text.tolist()
    queries = checked_df.synthetic_query.to_list()

    # initialise bm25 instances
    # trained on all documents
    bm25 = tokenize_corpus(sample_texts, bm25_algorithm)
    # trained on all queries
    bm25_queries = tokenize_corpus(sample_queries, bm25_algorithm)

    negatives_ = []
    if sample_technique == "random":
        for i, query in tqdm(enumerate(queries)):
            #  generate (text, score) information for all documents selected by the
            #  negative sampling technique
            negative_docs_scores = negative_sample_from_topK(
                query, bm25, negative_samples, sample_texts
            )
            # will be N document-score pairs
            negatives_.append(negative_docs_scores)
    elif sample_technique == "gold_batch":
        batches = []
        all_queries = sample_queries
        # for each query generate a batch containing similar queries
        for query in tqdm(queries):
            batch = generate_batches(query, all_queries, bm25_queries, batch_size)
            batches.append(batch)

        for batch in batches:
            main_query = batch[0]
            # generate list of 'gold documents' - documents which are most similar to
            # a query within the batch
            gold_docs = []
            for i in range(1, len(batch)):
                query = batch[i]
                # only return most similar (k=1)
                gold_doc, gold_score = bm25_search(query, bm25, 1)
                gold_docs.append(sample_texts[gold_doc[0]])
            # generate similarity scores for all texts in corpus to the query considered
            all_docs, all_scores = bm25_search(main_query, bm25, len(sample_texts))
            # filter to only return gold documents and their scores
            # (similarities to main query)
            scores = [
                (sample_texts[all_docs[a]], all_scores[a])
                for a in range(len(all_docs))
                if sample_texts[all_docs[a]] in gold_docs
            ]
            # sort in descending order and return top N most relevant gold documents
            scores = sorted(scores, key=lambda x: x[1], reverse=True)[
                : int(negative_samples)
            ]
            negatives_.append(scores)
    elif sample_technique == "keywords":
        # convert each document in the corpus to a list of keywords
        entities = []
        for i, doc in tqdm(enumerate(sample_texts)):
            # https://pypi.org/project/rake-nltk/
            rake_nltk_var = Rake()
            rake_nltk_var.extract_keywords_from_text(doc)
            # keywords are ordered by descending importance
            # want top 10 most important unique keywords
            keywords = list(dict.fromkeys(rake_nltk_var.get_ranked_phrases()))[:10]
            entities.append(keywords)

        for i, query in tqdm(enumerate(queries)):
            # get document originally paired with query
            paired_doc = paired_docs[queries.index(query)]
            rake_nltk_var = Rake()
            rake_nltk_var.extract_keywords_from_text(paired_doc)
            # want top 10 most important unique keywords
            keyword_extracted = list(dict.fromkeys(rake_nltk_var.get_ranked_phrases()))[
                :10
            ]

            out = []
            for doc in entities:
                # score all documents in corpus by how many keywords they have in common
                # with the paired document
                score = 0
                for j in range(len(keyword_extracted)):
                    if keyword_extracted[j] in doc:
                        # keywords are weighted - those earlier in list are most
                        # relevant to documents and therefore are highly scored
                        score += len(keyword_extracted) - j
                if score > 0:
                    #  disregard all documents which have no keywords in common
                    out.append((sample_texts[entities.index(doc)], score))

            # remove duplicate documents
            out = list(dict.fromkeys(out))
            # if data sufficient, there should be enough documents remaining to select
            #  negative samples
            if len(out) > negative_samples + 2:
                # discard top 2 documents (original document and closest document) so
                # negative samples are similar but not too similar to the original
                out = sorted(out, key=lambda x: x[1], reverse=True)[
                    2 : negative_samples + 2
                ]
                # need BM25 score between query and files for output JSON file
                all_docs, all_scores = bm25_search(query, bm25, len(sample_texts))
                formatted_out = []
                for sample, common in out:
                    idx = sample_texts.index(sample)
                    assert sample_texts[int(np.where(all_docs == idx)[0])] == sample
                    idx = int(np.where(all_docs == idx)[0])
                    assert all_docs[idx] == sample_texts.index(sample)
                    bm25_score = all_scores[idx]
                    formatted_out.append((sample, bm25_score))

                negatives_.append(formatted_out)
            else:
                # NOTE: if data is too sparse for this technique we revert to random
                # sampling for this query only
                print(
                    "The 'Keywords' negative sampling approach fails on this query \
                      due to data sparsity - reverting to random sample generation..."
                )
                #  generate (text, score) information for all documents selected by the
                #  negative sampling technique
                negative_docs_scores = negative_sample_from_topK(
                    query, bm25, negative_samples, sample_texts
                )
                # will be N document-score pairs
                negatives_.append(negative_docs_scores)
    else:
        print("Invalid negative sampling approach!")
        exit()

    # will end up with N lines in the final output for each original doc-query pair
    # one line for each negative sample chosen (N)
    checked_df["negative_sample"] = negatives_
    checked_df = checked_df.explode("negative_sample")
    checked_df[["negative_sample", "negative_bm25_score"]] = pd.DataFrame(
        checked_df["negative_sample"].tolist(), index=checked_df.index
    )
    return checked_df


# endregion

# region tasks


@task(
    cache=True,
    cache_version="1.0.3",
    interruptible=True,
    disable_deck=False,
    requests=Resources(cpu="2", mem="5Gi"),
)
def negative_sampling(
    checked_df: pd.DataFrame,
    full_df: pd.DataFrame,
    bm25_algorithm: str = "bm25",
    sample_technique: str = "random",
    batch_size: int = 32,
    negative_samples: int = 2,
    number_documents: int = 32,
) -> Annotated[pd.DataFrame, TopFrameRenderer(10)]:
    """task function for negative sampling.

    Args:
        checked_df (DataFrame): Pandas dataframe output from consistency check
        full_df (DataFrame): Dataframe containing all queries and docs in corpus
        bm25_algorithm (str): bm25 algorithm to use, can be:
        either 'bm25', 'bm25l', or 'bm25plus'
        sample_technique (str): sampling technique to use, can be:
        either 'random', 'gold_batch', or 'keywords'
        batch_size (int): batch size for batch-based negative sampling techniques
        negative_samples (int): number of documents to choose as negative
        samples for a query
        number_documents (int): number of pairs to consider
    """
    if sample_technique == "keywords":
        nltk.download("stopwords")
        nltk.download("punkt")

    if sample_technique != "gold_batch":
        batch_size = None

    return negative_sampling_bm25(
        checked_df=checked_df,
        full_df=full_df,
        bm25_algorithm=bm25_algorithm,
        sample_technique=sample_technique,
        batch_size=batch_size,
        negative_samples=negative_samples,
        number_documents=number_documents,
    )


# endregion
