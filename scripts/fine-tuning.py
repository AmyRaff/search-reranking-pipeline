import shutil
import pandas as pd
import logging
import os
from torch.utils.data import DataLoader
import torch
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers import InputExample

from optimum.onnxruntime import ORTQuantizer, ORTModelForSequenceClassification
from optimum.onnxruntime.configuration import AutoQuantizationConfig

import transformers
from transformers.onnx import FeaturesManager
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
import json

DATA_OUT_DIR = os.environ.get(
    "DATA_OUT_PATH",
    "pfs/out/",
)

MODELS = os.environ.get(
    "MODELS",
    '["cross-encoder/ms-marco-TinyBERT-L-2-v2"]',
)

NUMBER_DOCUMENTS = os.environ.get(
    "NUMBER_DOCUMENTS",
    "20",
    # NOTE: small dataset used for checking, set to -1 for full dataset
)

BATCH_SIZE = os.environ.get(
    "BATCH_SIZE",
    "64",
)

NUM_EPOCHS = os.environ.get(
    "NUM_EPOCHS",
    "2",
)

MAX_LENGTH = os.environ.get(
    "MAX_LENGTH",
    "100",
    # NOTE: max length of input sequences to consider, don't change
)

TRAINING_DATA_DIR = os.environ.get(
    "TRAINING_DATA_DIR",
    "data/negative_sampled/synthesis",
)


def load_training_samples(training_file):
    """Reads a training file and generates a list of training data in the form of
    document-query pairs and their positive/negative labels.

    Args:
        training_file (str): Name of file to use as training data. Will be output of
        negative sampling.

    Returns:
        List[str]: list of document-query pairs and their positive/negative labels
        used as training data
    """
    logger.info("reading training data")
    # read file and convert to dataframe
    training_filepath = os.path.join(TRAINING_DATA_DIR, training_file)
    df = pd.read_json(training_filepath, lines=True)
    df.index = range(len(df))

    if NUMBER_DOCUMENTS != -1:
        # optional filtering to only test with a small subset of documents
        # NOTE: otherwise uses full file
        df = df.iloc[:NUMBER_DOCUMENTS]
        logger.info(f"test run with {NUMBER_DOCUMENTS} documents")

    logger.info("processing training data")
    # positive examples are duplicated (or more) due to how data is previously generated
    # and stored
    positive_examples = [
        InputExample(texts=[row["synthetic_query"], row["text"]], label=1)
        for _, row in df.drop_duplicates("text").iterrows()
    ]
    # negative examples on the contrary are not
    negative_examples = [
        InputExample(texts=[row["synthetic_query"], row["negative_sample"]], label=0)
        for _, row in df.iterrows()
    ]
    #  build training set
    train_samples = positive_examples + negative_examples

    return train_samples


def onnx_export(model_path, logger):
    """Exports a model to ONNX format.

    Args:
        model_path (str): Path to model to quantise
        logger (logger): logger for command line outputs
    """
    logger.info("exporting model to onnx runtime")
    # load model and tokenizer
    logger.debug("loading model into transformers library format")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # load config
    model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(
        model, feature="sequence-classification"
    )
    onnx_config = model_onnx_config(model.config)
    # export
    logger.info("starting export")
    onnx_inputs, onnx_outputs = transformers.onnx.export(
        preprocessor=tokenizer,
        model=model,
        config=onnx_config,
        opset=13,
        output=Path(os.path.join(model_path, "model.onnx")),
    )
    logger.info(f"model exported at {model_path}/model.onnx")
    return onnx_inputs, onnx_outputs


def quantise_model(model_path, logger):
    """Quantises a model and returns its path.

    Args:
        model_path (str): Path to model to quantise
        logger (logger): logger for command line outputs

    Returns:
        str: Path of resulting quantised model
    """
    logger.info("quantising model")
    # Load PyTorch model and convert to ONNX
    onnx_model = ORTModelForSequenceClassification.from_pretrained(
        os.path.join(model_path)
    )
    # Create quantizer
    quantizer = ORTQuantizer.from_pretrained(onnx_model)
    # Define the quantization strategy by creating the appropriate configuration
    dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    # Quantize the model
    model_quantized_path = quantizer.quantize(
        save_dir=model_path,
        quantization_config=dqconfig,
    )
    return model_quantized_path


def fine_tune_crossencoder(
    training_file,
    model_name,
    consistency_check_type,
    negative_sampling_type,
    device,
    logger,
):
    logger.info(f"entering fine-tuning process using {training_file}")
    # generate list of doc-query pairs and their positive/negative labels
    train_samples = load_training_samples(training_file)
    # format training data into shuffled batches
    train_dataloader = DataLoader(
        train_samples, shuffle=True, batch_size=64, drop_last=True
    )
    logger.info("training data processed")

    num_labels = 1
    max_length = MAX_LENGTH  # max length of input sequences to consider

    logger.info(f"loading cross-encoder model {model_name}")
    #  build crossencoder
    model = CrossEncoder(
        model_name,
        num_labels=num_labels,
        max_length=max_length,
        default_activation_function=torch.nn.Identity(),
        device=device,
    )
    logger.info(f"working on {device}")

    # warmup_steps = 96
    lr = 4e-7
    logger.info("fitting the model on training data (fine-tuning)")
    # model training
    model.fit(
        train_dataloader=train_dataloader,
        epochs=NUM_EPOCHS,
        optimizer_params={"lr": lr},
        use_amp=True,
    )
    logger.debug("fit completed")

    # name and save model
    model_out_path = (
        DATA_OUT_DIR
        + model_name.replace("/", "___")
        + "_"
        + consistency_check_type
        + "_"
        + negative_sampling_type
        + "_"
        + "model"
    )
    # debugging
    if os.path.exists(model_out_path) and os.path.isdir(model_out_path):
        shutil.rmtree(model_out_path)

    model.save(model_out_path)
    logger.info(f"model saved at {model_out_path}")

    # export model to ONNX format
    onnx_export(model_out_path, logger)
    # quantise model
    quantise_model(model_out_path, logger)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    NUMBER_DOCUMENTS = int(NUMBER_DOCUMENTS)
    BATCH_SIZE = int(BATCH_SIZE)
    NUM_EPOCHS = int(NUM_EPOCHS)
    MAX_LENGTH = int(MAX_LENGTH)
    models = json.loads(MODELS)
    for training_file in sorted(os.listdir(TRAINING_DATA_DIR)):
        # check using relevant files only
        if training_file.endswith(".jsonl"):
            for model_name in models:
                consistency_check_type = training_file.split("_merged")[0]
                negative_sampling_type = training_file.split("negative_samples_")[
                    1
                ].split(".")[0]
                logger = logging.getLogger(
                    f"cross-encoder fine-tuning - {model_name} - \
                        {consistency_check_type} - {negative_sampling_type}"
                )
                logger.info(" Starting fine-tuning process")
                # creates a finetuned model for all consistency check methods
                # and negative sampling methods (all outputs of negative_sampling.py)
                fine_tune_crossencoder(
                    training_file=training_file,
                    model_name=model_name,
                    consistency_check_type=consistency_check_type,
                    negative_sampling_type=negative_sampling_type,
                    logger=logger,
                    device=device,
                )
