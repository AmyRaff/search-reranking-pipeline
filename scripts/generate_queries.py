import pandas as pd
from datetime import datetime
import logging
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from tqdm_loggable.auto import tqdm
from torch.utils.data import Dataset
import torch

DATA_PATH = os.environ.get(
    "DATA_PATH",
    "./data/data_to_use_Feb2022/sample_synthesis.jsonl",
)

DATA_OUT_DIR = os.environ.get(
    "DATA_OUT_PATH",
    "pfs/out/",
)

TEXT_COLUMN = os.environ.get(
    "TEXT_COLUMN",
    "summary_sentences",
)

MODEL = os.environ.get(
    "MODEL",
    "google/flan-t5-xl",
)

NUMBER_DOCUMENTS = os.environ.get(
    "NUMBER_DOCUMENTS",
    "20",
)

BATCH_SIZE = os.environ.get(
    "BATCH_SIZE",
    "4",
)

PROMPT_PATH = os.environ.get(
    "PROMPT_PATH",
    "configs/prompts/synthesis/",
)


class ListDataset(Dataset):
    def __init__(self, original_list):
        self.original_list = original_list

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, i):
        return self.original_list[i]


def main(prompt_file, pipeline, logger):
    prompt_filepath = os.path.join(PROMPT_PATH, prompt_file)
    with open(prompt_filepath, "r") as fp:
        prompt = fp.read()

    df = pd.read_json(DATA_PATH, lines=True)
    df["text"] = df[TEXT_COLUMN].str.join(" ").str.strip()

    df.index = range(len(df))

    if NUMBER_DOCUMENTS != -1:
        df = df.iloc[:NUMBER_DOCUMENTS]

    texts = df.text.apply(lambda x: prompt.replace("<< DOCUMENT >>", x)).tolist()

    start = datetime.now()
    logger.info("started generation at: %s" % start)
    logger.info("Number of documents: %s" % len(df))

    dataset = ListDataset(texts)

    start = datetime.now()
    res = []
    for out in tqdm(pipeline(dataset)):
        res += out
    print(datetime.now() - start)

    df["synthetic_query"] = [x["generated_text"] for x in res]

    logger.info("generation lasted: %s" % (datetime.now() - start))

    data_out_path = (
        DATA_OUT_DIR
        + DATA_PATH.split("/")[-1].split(".")[0]
        + "_"
        + prompt_file.split(".tx")[0]
        + ".jsonl"
    )
    df.to_json(data_out_path, lines=True, orient="records")

    return


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("query_generation_synthesis")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"working on {device}")

    NUMBER_DOCUMENTS = int(NUMBER_DOCUMENTS)
    BATCH_SIZE = int(BATCH_SIZE)

    tokenizer = T5Tokenizer.from_pretrained(MODEL)
    model = T5ForConditionalGeneration.from_pretrained(MODEL, device_map="auto")

    pipe = pipeline(
        task="text2text-generation",
        model=model,
        tokenizer=tokenizer,
        batch_size=BATCH_SIZE,
        device=device,
        device_map="auto",
    )

    for prompt_file in os.listdir(PROMPT_PATH):
        # actually one file due to how we mount repos in pachyderm
        logger.info(f"working with prompt {prompt_file}")
        main(prompt_file, pipeline=pipe, logger=logger)
