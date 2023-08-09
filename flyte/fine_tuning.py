import pandas as pd
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
from flytekit import task, Resources, current_context
import joblib
from flytekit.types.file import JoblibSerializedFile


# region utils


def load_training_samples(df, number_documents):
    """Reads a dataframe and generates a list of training data in the form of
    document-query pairs and their positive/negative labels.

    Args:
        df (DataFrame): Training data taken from  output of negative sampling.

    Returns:
        List[str]: list of document-query pairs and their positive/negative labels
        used as training data
    """

    # format dataframe
    df.index = range(len(df))

    if number_documents != -1:
        # optional filtering to only test with a small subset of documents
        # NOTE: otherwise uses full file
        df = df.iloc[:number_documents]

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


def onnx_export(model_path):
    """Exports a model to ONNX format.

    Args:
        model_path (str): Path to model to quantise
    """
    # load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # load config
    model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(
        model, feature="sequence-classification"
    )
    onnx_config = model_onnx_config(model.config)
    # export
    onnx_inputs, onnx_outputs = transformers.onnx.export(
        preprocessor=tokenizer,
        model=model,
        config=onnx_config,
        opset=13,
        output=Path(os.path.join(model_path, "model.onnx")),
    )
    return onnx_inputs, onnx_outputs


def quantise_model(model_path):
    """Quantises a model and returns its path.

    Args:
        model_path (str): Path to model to quantise

    Returns:
        str: Path of resulting quantised model
    """
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


# endregion

# region main-code


@task(
    cache=False,
    cache_version="1.0.3",
    interruptible=True,
    disable_deck=False,
    requests=Resources(cpu="2", mem="5Gi"),
)
def fine_tune_crossencoder(
    training_df: pd.DataFrame,
    model_name: str,
    consistency_check_type: str,
    negative_sampling_type: str,
    max_length: int = 100,
    num_epochs: int = 2,
    number_documents: int = 32,
) -> JoblibSerializedFile:
    """

    Args:
        training_df (DataFrame): Pandas dataframe output from negative sampling
        model_name (str): path of the model to use for the cross encoder
        consistency_check_type (str): the type of consistency check used
        negative_sampling_type (str): the type of negative sampling used
        max_length (int): max length of input sequences to consider, NOTE: don't change
        num_epochs (int): number of training epochs
        number_documents (int): number of pairs to consider
    """

    # generate list of doc-query pairs and their positive/negative labels
    train_samples = load_training_samples(training_df, number_documents)
    # format training data into shuffled batches
    train_dataloader = DataLoader(
        train_samples, shuffle=True, batch_size=64, drop_last=True
    )

    num_labels = 1

    #  build crossencoder
    model = CrossEncoder(
        model_name,
        num_labels=num_labels,
        max_length=max_length,
        default_activation_function=torch.nn.Identity(),
    )

    # warmup_steps = 96
    lr = 4e-7
    # model training
    model.fit(
        train_dataloader=train_dataloader,
        epochs=num_epochs,
        optimizer_params={"lr": lr},
        use_amp=True,
    )

    working_dir = current_context().working_directory
    # name and save model
    model_out_path = (
        model_name.replace("/", "___")
        + "_"
        + consistency_check_type
        + "_"
        + negative_sampling_type
        + "_"
        + "model.joblib.dat"
    )
    model_out_path = os.path.join(working_dir, model_out_path)

    joblib.dump(model, model_out_path)
    # return the model in a format accepted by flyte
    return JoblibSerializedFile(path=model_out_path)

    # NOTE: can return model in different formats if needed (unsure)
    # export model to ONNX format
    # onnx_export(model_out_path, logger)
    # quantised model
    # quantise_model(model_out_path, logger)


# endregion
