import pandas as pd
import logging
import glob
import json
import os

DATA_OUT_DIR = os.environ.get(
    "DATA_OUT_PATH",
    "pfs/out/",
)

QUERIES_PATH = os.environ.get(
    "QUERIES_PATH",
    "data/consistency_checked/synthesis/",
)

CONSISTENCY_CHECKS_LIST = os.environ.get(
    "CONSISTENCY_CHECKS_LIST",
    '["bm25", "bm25plus", "bm25l", "crossencoder", "combined_filter_bm25_crossencoder",\
    "combined_filter_bm25l_crossencoder", "combined_filter_bm25plus_crossencoder", \
    "combined_filter_bm25_linear", "combined_filter_bm25l_linear", \
    "combined_filter_bm25plus_linear", "combined_search_bm25", "combined_search_bm25l",\
    "combined_search_bm25plus"]',
)


def merge_prompts(queries_path, logger):
    """Function to merge information from multiple JSONL files into one file.

    Args:
        queries_path (str): path to look for files to merge. Will contain glob filtering
        logger (logger): logger for command line outputs
    """
    # get all files which satisfy the glob constraints - should be 6 files
    queries_files = sorted(glob.glob(queries_path))
    # NOTE: this might change?
    assert len(queries_files) == 6

    # merge JSONL files into one JSONL file
    consistency_check_dfs = {}
    logger.info(f"prompt results to merge: {queries_files}")
    for file in queries_files:
        consistency_check_dfs[int(file.split("_checked_")[0][-1])] = pd.read_json(
            file, lines=True
        )
    logger.info("merging results...")
    df = consistency_check_dfs[1].copy()
    df["prompt"] = 1

    if len(consistency_check_dfs) == 1:
        return df

    for prompt in sorted(consistency_check_dfs.keys())[1:]:
        tmp = consistency_check_dfs[prompt].copy()
        tmp["prompt"] = prompt
        df = pd.concat((df, tmp))

    logger.info("dropping duplicate generated queries")
    # remove duplicates
    df = df.drop_duplicates(["text", "synthetic_query"])
    logger.info(f"final dataset has {len(df)} (document,query) pairs")
    # generate output dataframe
    data_out_path = (
        DATA_OUT_DIR
        + file.split("_checked_")[1].split(".json")[0]
        + f"_merged1to{len(queries_files)}"
        + ".jsonl"
    )
    df.to_json(data_out_path, lines=True, orient="records")
    logger.info(f"merged results saved at {data_out_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    consistency_checks_list = json.loads(CONSISTENCY_CHECKS_LIST)
    for consistency_check in consistency_checks_list:
        # NOTE: changed naming conventions here to allow for new files
        queries_path = QUERIES_PATH + "*" + f"_checked_{consistency_check}.jsonl" + "*"
        logger = logging.getLogger(f"merging prompts results - {consistency_check}")
        merge_prompts(queries_path, logger=logger)
