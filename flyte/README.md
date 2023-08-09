# Flyte Pipeline Documentation

## Table of Contents

- [Building a Pipeline](#building-a-pipeline)
- [Consistency Check Algorithms](#consistency-check-algorithms)
- [Negative Sampling Algorithms](#negative-sampling-algorithms)


## Building a Pipeline

The required workflow structure for this project can be found in **pipeline_workflow.py** and contains 4 stages:

- Load Input Data (texts and their associated synthetic queries, along with metadata)
- Consistency Check Algorithm
- Negative Sampling Algorithm
- Finetune and Output Resulting Model 

During the intern project we have experimented with alternative consistency check and negative sampling approaches, all of which are implemented in this repository, and available for use with the pipeline.

Further details on the approaches we have implemented can be found on Confluence - [Consistency Checks](https://adarga.atlassian.net/wiki/spaces/DSRN/pages/3787259912/Consistency+Check+Experiments) and [Negative Sampling](https://adarga.atlassian.net/wiki/spaces/DSRN/pages/3797057594/Negative+Sampling+Experiments).

---

## Consistency Check Algorithms

There are 13 various consistency checking approaches implemented in this code, all accessible through the following methods and parameter customisations. These variations are:

- BM25
- BM25L
- BM25Plus
- Cross-Encoder
- Search Combination with BM25
- Search Combination with BM25L
- Search Combination with BM25Plus
- Filtering Combination with BM25 and Linear Scoring
- Filtering Combination with BM25L and Linear Scoring
- Filtering Combination with BM25Plus and Linear Scoring
- Filtering Combination with BM25 and CrossEncoder Scoring
- Filtering Combination with BM25L and CrossEncoder Scoring
- Filtering Combination with BM25Plus and CrossEncoder Scoring

Details can be found on [Confluence](https://adarga.atlassian.net/wiki/spaces/DSRN/pages/3787259912/Consistency+Check+Experiments).

### BM25-Based

This is the basic BM25 implementation, which uses the base algorithm with no added steps.

This approach can be used with BM25, BM25L and BM25Plus.

    df_consistency_check = consistency_check_bm25_based(
        df=df,
        number_documents=32,
        bm25_algorithm="bm25"
    )

#### Customisable Parameters:

- **number_documents** = Number of documents to apply the algorithm to. Small values used for testing, **use -1 to apply to an entire file**.

- **bm25_algorithm** = Type of algorithm to use. Can be **BM25**, **BM25L** or **BM25Plus**.


### Cross-Encoder

This is the basic cross-encoder implementation, with no added steps.

    df_consistency_check = consistency_check_crossencoder(
        df=df,
        batch_size=16,
        number_documents=32,
        cross_encoder_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
        TOP_K=10000,
    )

#### Customisable Parameters:

- **batch_size** = Batch size for training the cross-encoder.

- **number_documents** = Number of documents to apply the algorithm to. Small values used for testing, **use -1 to apply to an entire file**.

- **cross_encoder_name** = Path to cross-encoder used.

- **TOP_K** = Number of most similar (highest scoring) document-query pairs returned by the cross-encoder. Can be changed in future but 10,000 performs well. See Confluence for further details.

### Combined Search 

This is the combined approach which utilises both BM25-based algorithms and the cross-encoder in the search step. More details on the algorithm can be found [here](https://adarga.atlassian.net/wiki/spaces/DSRN/pages/3787259912/Consistency+Check+Experiments#(3)-BM25-and-Cross-encoder-Combination-for-Search).

    df_consistency_check = consistency_check_combined_search(
        df=df,
        batch_size=16,
        number_documents=32,
        bm25_algorithm="bm25",
        cross_encoder_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
    )

#### Customisable Parameters:

- **batch_size** = Batch size for training the cross-encoder.

- **number_documents** = Number of documents to apply the algorithm to. Small values used for testing, **use -1 to apply to an entire file**.

- **bm25_algorithm** = Type of algorithm to use. Can be **BM25**, **BM25L** or **BM25Plus**.

- **cross_encoder_name** = Path to cross-encoder used.

### Combined Filtering

This is the combined approach which utilises BM25-based algorithms for document search and then uses the cross-encoder to filter the results. More details on the algorithm can be found [here](https://adarga.atlassian.net/wiki/spaces/DSRN/pages/3787259912/Consistency+Check+Experiments#(1)-BM25-and-Cross-encoder-Combination-for-Filtering).

    df_consistency_check = consistency_check_combined_filtering(
        df=df,
        batch_size=16,
        number_documents=32,
        scoring_method="linear",
        bm25_algorithm="bm25l",
        cross_encoder_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
        TOP_K=10000,
    )

#### Customisable Parameters

- **batch_size** = Batch size for training the cross-encoder.

- **number_documents** = Number of documents to apply the algorithm to. Small values used for testing, **use -1 to apply to an entire file**.

- **scoring_method** = Relevance scoring method to use. Can be **Linear** or **CrossEncoder**. Details on the differences can be found on Confluence.

- **bm25_algorithm** = Type of algorithm to use. Can be **BM25**, **BM25L** or **BM25Plus**.

- **cross_encoder_name** = Path to cross-encoder used.

- **TOP_K** = Number of most similar (highest scoring) document-query pairs returned by the cross-encoder. Can be changed in future but 10,000 performs well. See Confluence for further details.

---

## Negative Sampling Algorithms

There are 7 different negative sampling approaches implemented in this code, all accessible through the following methods and parameter customisations. These variations are:

- Random Sampling with BM25
- Random Sampling with BM25L
- Random Sampling with BM25Plus
- Gold Batch Sampling with BM25
- Gold Batch Sampling with BM25L
- Gold Batch Sampling with BM25Plus
- Keyword-Based Sampling

Details can be found on [Confluence](https://adarga.atlassian.net/wiki/spaces/DSRN/pages/3797057594/Negative+Sampling+Experiments).

There is only one pipeline function, instead of multiple, for negative sampling. The sampling approach used fully depends on the **sample_technique** parameter.

### Random Sampling

**sample_technique = 'random'**

This is a simple approach which generates negative samples by randomly sampling the 6th to 50th most relevant documents to a query. These values were decided empirically and perform very well but can be changed within the code if needed.

    df_negative = negative_sampling(
        checked_df=df_consistency_check,
        full_df=df,
        bm25_algorithm="bm25l",
        sample_technique="random",
        batch_size=None,
        negative_samples=2,
        number_documents=32,
    )

#### Customisable Parameters:

- **bm25_algorithm** = Type of algorithm to use. Can be **BM25**, **BM25L** or **BM25Plus**.

- **negative_samples** = Number of negative samples to generate for each query.

- **number_documents** = Number of documents to apply the algorithm to. Small values used for testing, **use -1 to apply to an entire file**.


### Gold Batch Sampling

**sample_technique = 'gold_batch'**

A slightly more complicated approach which involves splitting queries into batches, and then sampling from the 'gold documents' of the inter-batch queries. The algorithm is explained in detail [here](https://adarga.atlassian.net/wiki/spaces/DSRN/pages/3797057594/Negative+Sampling+Experiments#BM25-and-Gold-Batch-Negative-Sampling).

    df_negative = negative_sampling(
        checked_df=df_consistency_check,
        full_df=df,
        bm25_algorithm="bm25",
        sample_technique="gold_batch",
        batch_size=32,
        negative_samples=2,
        number_documents=32,
    )

#### Customisable Parameters:

- **bm25_algorithm** = Type of algorithm to use. Can be **BM25**, **BM25L** or **BM25Plus**.

- **batch_size** = Batch size for query-batching. Explained on [Confluence](https://adarga.atlassian.net/wiki/spaces/DSRN/pages/3797057594/Negative+Sampling+Experiments#BM25-and-Gold-Batch-Negative-Sampling).

- **negative_samples** = Number of negative samples to generate for each query.

- **number_documents** = Number of documents to apply the algorithm to. Small values used for testing, **use -1 to apply to an entire file**.


### Keyword-Based Sampling

**sample_technique = 'keywords'**

A negative sampling approach which focuses on comparing the results of keyword extraction instead of relying on any BM25-based algorithm. The process is explained in detail [here](https://adarga.atlassian.net/wiki/spaces/DSRN/pages/3797057594/Negative+Sampling+Experiments#Keyword-Based-Negative-Sampling).


    df_negative = negative_sampling(
        checked_df=df_consistency_check,
        full_df=df,
        bm25_algorithm="bm25",
        sample_technique="keywords",
        batch_size=None,
        negative_samples=2,
        number_documents=32,
    )

#### Customisable Parameters:

**NOTE**: In the event of this technique failing (see [Confluence](https://adarga.atlassian.net/wiki/spaces/DSRN/pages/3797057594/Negative+Sampling+Experiments#Keyword-Based-Negative-Sampling)) for a given doc-query pair, it reverts to random sampling, which utilises BM25-based algorithms.

- **bm25_algorithm** = Type of algorithm to use. Can be **BM25**, **BM25L** or **BM25Plus**.

- **negative_samples** = Number of negative samples to generate for each query.

- **number_documents** = Number of documents to apply the algorithm to. Small values used for testing, **use -1 to apply to an entire file**.