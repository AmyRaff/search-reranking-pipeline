from flyte.utils import get_data, load_data
from flyte.consistency_check import (
    consistency_check_crossencoder,
    consistency_check_bm25_based,
    consistency_check_combined_filtering,
    consistency_check_combined_search,
)
from flyte.negative_sampling import negative_sampling
from flyte.fine_tuning import fine_tune_crossencoder
from flytekit import workflow
from flytekit.types.file import JoblibSerializedFile

#  models to use for training
MODEL = "cross-encoder/ms-marco-TinyBERT-L-2-v2"


@workflow
def verbose_wf() -> JoblibSerializedFile:
    dir = get_data()
    df = load_data(dir=dir)
    df_consistency_check = consistency_check_crossencoder(
        df=df,
        batch_size=16,
        number_documents=32,
        cross_encoder_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
        TOP_K=10000,
    )
    df_consistency_check = consistency_check_bm25_based(
        df=df,
        number_documents=32,
        bm25_algorithm="bm25l",
    )
    df_consistency_check = consistency_check_combined_search(
        df=df,
        batch_size=16,
        number_documents=32,
        bm25_algorithm="bm25",
        cross_encoder_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
    )
    df_consistency_check = consistency_check_combined_filtering(
        df=df,
        batch_size=16,
        number_documents=32,
        scoring_method="linear",
        bm25_algorithm="bm25l",
        cross_encoder_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
        TOP_K=10000,
    )
    df_consistency_check = consistency_check_combined_filtering(
        df=df,
        batch_size=16,
        number_documents=32,
        scoring_method="crossencoder",
        bm25_algorithm="bm25plus",
        cross_encoder_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
        TOP_K=10000,
    )

    df_negative = negative_sampling(
        checked_df=df_consistency_check,
        full_df=df,
        bm25_algorithm="bm25l",
        sample_technique="random",
        batch_size=None,
        negative_samples=2,
        number_documents=32,
    )
    df_negative = negative_sampling(
        checked_df=df_consistency_check,
        full_df=df,
        bm25_algorithm="bm25",
        sample_technique="gold_batch",
        batch_size=32,
        negative_samples=2,
        number_documents=32,
    )
    df_negative = negative_sampling(
        checked_df=df_consistency_check,
        full_df=df,
        bm25_algorithm="bm25plus",
        sample_technique="keywords",
        batch_size=None,
        negative_samples=2,
        number_documents=32,
    )

    out_model = fine_tune_crossencoder(
        training_df=df_negative,
        model_name=MODEL,
        consistency_check_type="consistency_check_bm25",
        negative_sampling_type="negative_sampling_keywords",
        max_length=100,
        num_epochs=2,
        number_documents=32,
    )

    df >> df_consistency_check
    df_consistency_check >> df_negative
    df_negative >> out_model

    return out_model


@workflow
def example_wf() -> JoblibSerializedFile:
    dir = get_data()
    df = load_data(dir=dir)

    df_consistency_check = consistency_check_combined_filtering(
        df=df,
        batch_size=16,
        number_documents=32,
        scoring_method="crossencoder",
        bm25_algorithm="bm25plus",
        cross_encoder_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
        TOP_K=10000,
    )

    df_negative = negative_sampling(
        checked_df=df_consistency_check,
        full_df=df,
        bm25_algorithm="bm25l",
        sample_technique="random",
        batch_size=None,
        negative_samples=2,
        number_documents=32,
    )

    out_model = fine_tune_crossencoder(
        training_df=df_negative,
        model_name=MODEL,
        consistency_check_type="consistency_check_bm25",
        negative_sampling_type="negative_sampling_keywords",
        max_length=100,
        num_epochs=2,
        number_documents=32,
    )

    df >> df_consistency_check
    df_consistency_check >> df_negative
    df_negative >> out_model

    return out_model
