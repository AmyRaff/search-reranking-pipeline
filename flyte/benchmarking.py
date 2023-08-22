import glob
import pandas as pd
import torch
from sentence_transformers.cross_encoder import CrossEncoder
from ranx import Qrels, Run, compare
import json
import jsonlines
import os
from numpy import exp
from flytekit import task, Resources, current_context, workflow
from typing_extensions import Annotated
from flytekit.deck.renderer import TopFrameRenderer

MODELS = '["cross-encoder/ms-marco-MiniLM-L-6-v2"]'

# MODELS_DIR = os.environ.get("MODELS_DIR", "fine-tuned-models/")

# BENCHMARKING_DATA = os.environ.get(
#     "BENCHMARKING_DATA",
#     (
#         "pfs/out/"
#         + "search_results_events_with_relevance_score_results_merged1to5.jsonl"
#     ),
# )


def sigmoid(x):
    return 1.0 / (1.0 + exp(-x))


def load_benchmarking_data(df):
    """Loads the benchmarking dataframe into the desired data formats.

    Args:
        df (pd.DataFrame): benchmarking data to be loaded.

    Returns:
        pd.DataFrame, Qrels, Qrels: formatted dataframe and data structures.
    """
    # format dataframe
    df.index = range(len(df))
    df.query_id = df.query_id.astype(str)
    df["text_id"] = df.text.apply(lambda x: hash(x))
    df.text_id = df.text_id.astype(str)
    df.relevance_score -= 1
    df.relevance_score = df.relevance_score.astype(int)

    # get binary benchmarking data
    df_binary = df.copy()
    df_binary.relevance_score = (df.score > 1).astype(int)

    # build benchmarking data structures
    qrels = Qrels.from_df(
        df=df,
        q_id_col="query_id",
        doc_id_col="text_id",
        score_col="relevance_score",
    )

    qrels_binary = Qrels.from_df(
        df=df_binary,
        q_id_col="query_id",
        doc_id_col="text_id",
        score_col="relevance_score",
    )

    return df, qrels, qrels_binary


def load_models(models_list):
    """Loads a list of models from their filenames.

    Args:
        models_list (List[str]): List of model paths to benchmark.

    Returns:
        List[CrossEncoder]: List of model objects.
    """
    num_labels = 1
    max_length = 512

    models = []
    for model_name in models_list:
        models.append(
            CrossEncoder(
                model_name,
                num_labels=num_labels,
                max_length=max_length,
                default_activation_function=torch.nn.Identity(),
            )
        )

    return models


def cross_encoder_reranker(df, cross_encoder, model_name):
    """Reranks input data according to a specified CrossEncoder model.

    Args:
        df (pd.DataFrame): Data to rerank.
        cross_encoder (CrossEncoder): Model used.
        model_name (str): Name of model used.

    Returns:
        pd.DataFrame: Reranked data.
    """
    query_passage_pairs = [[x[1]["query"], x[1]["text"]] for x in df.iterrows()]
    cross_encoder_scores = cross_encoder.predict(query_passage_pairs)
    cross_encoder_scores = sigmoid(cross_encoder_scores)

    df[model_name] = cross_encoder_scores
    df[model_name] = df[model_name].astype(float)

    return df


def save_scores_report(report, path):
    """Saves a Qrels report object to the specified path.

    Args:
        report (Report): Metrics report to save.
        path (str): Filepath to save report to.
    """
    report = report.to_dict()
    for name in report["model_names"]:
        tmp = dict()
        tmp["name"] = name
        for metric, value in report[name]["scores"].items():
            tmp[metric] = value
        with jsonlines.open(path, mode="a") as writer:
            writer.write(tmp)


@task(
    cache=True,
    cache_version="1.0.3",
    interruptible=True,
    disable_deck=False,
    requests=Resources(cpu="2", mem="5Gi"),
)
def main(
    benchmarking_data: pd.DataFrame, models_list: list[str]
) -> Annotated[pd.DataFrame, TopFrameRenderer(10)]:
    """Main benchmarking functionality. Generates metrics and reports for
    all models.

    Args:
        benchmarking_data (pd.DataFrame): Benchmarking data to use.
        models_list (List[str]): List of model paths to benchmark.

    Returns:
        pd.DataFrame: Reranked data. Not used just there for consistency.
    """
    # load annotated benchmarking data
    df, qrels, qrels_binary = load_benchmarking_data(benchmarking_data)

    # load models
    models = load_models(models_list)

    # rerank search results
    for model, model_name in zip(models, models_list):
        df = cross_encoder_reranker(df, model, model_name)

    # load reranking results in data structures
    runs_dict = dict()
    runs_dict["current_system"] = Run.from_df(
        df=df,
        q_id_col="query_id",
        doc_id_col="text_id",
        score_col="score",
    )
    runs_dict["current_system"].name = "current_system"

    for model_name in models_list:
        runs_dict[model_name] = Run.from_df(
            df=df,
            q_id_col="query_id",
            doc_id_col="text_id",
            score_col=model_name,
        )
        runs_dict[model_name].name = model_name

    # run benchmarking metrics - graded
    report = compare(
        qrels=qrels,
        runs=sorted(list(runs_dict.values()), key=lambda x: x.name),
        metrics=["ndcg_burges@3", "ndcg_burges@5", "ndcg_burges@10", "ndcg_burges@20"],
        max_p=0.01,  # P-value threshold
    )

    # run benchmarking metrics - binary
    report_binary = compare(
        qrels=qrels,
        runs=sorted(list(runs_dict.values()), key=lambda x: x.name),
        metrics=[
            "map@3",
            "map@5",
            "map@10",
            "map@20",
            "mrr@3",
            "mrr@5",
            "mrr@10",
            "mrr@20",
            "recall@3",
            "recall@5",
            "recall@10",
            "recall@20",
            "precision@1",
            "precision@3",
            "precision@5",
            "precision@10",
            "precision@20",
            "f1@3",
            "f1@5",
            "f1@10",
            "f1@20",
        ],
        max_p=0.01,  # P-value threshold
    )

    # save all results
    working_dir = current_context().working_directory

    report.save(
        os.path.join(working_dir, "full_benchmarking_metrics_report_graded.json")
    )
    report_binary.save(
        os.path.join(working_dir, "full_benchmarking_metrics_report_binary.json")
    )

    # save results - only metrics
    save_scores_report(
        report, os.path.join(working_dir, "benchmarking_metrics_report_graded.jsonl")
    )
    save_scores_report(
        report_binary,
        os.path.join(working_dir, "benchmarking_metrics_report_binary.jsonl"),
    )

    # NOTE: this output isn't actually used, we're only interested in the saved reports!
    return df


@workflow
def wf() -> Annotated[pd.DataFrame, TopFrameRenderer(10)]:
    # load original crossencoder
    additional_models = json.loads(MODELS)

    # want all fine-tuned models and the originl crossencoder
    # TODO: point to directory containing all finetuned models (fine_tuning.py output)
    models_dir = ""
    models_list = glob.glob(models_dir + "*") + additional_models

    # generate benchmarking metrics reports for each model
    # TODO: point to benchmarking data (ask alberto)
    benchmarking_data = ""

    main(benchmarking_data=benchmarking_data, models_list=models_list)
