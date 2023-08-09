from flytekit import task, Resources
from flytekit.types.file import FlyteFile
import pandas as pd
from flytekit.deck.renderer import TopFrameRenderer

from typing_extensions import Annotated


@task(cache=True, cache_version="1.0.0", interruptible=True)
def get_data() -> FlyteFile:
    # This file is a parquet file stored in S3
    return FlyteFile("s3://adarga-ds-dvc/files/md5/5f/992160d545b77dbeae16875e360523")


@task(cache=True, cache_version="1.1.0", disable_deck=False, interruptible=True)
def load_data(dir: FlyteFile) -> Annotated[pd.DataFrame, TopFrameRenderer(10)]:
    path = dir.download()
    df = pd.read_parquet(path)
    return df


@task(interruptible=True, disable_deck=False, requests=Resources(cpu="2", mem="5Gi"))
def truncate_length(df: pd.DataFrame) -> Annotated[pd.DataFrame, TopFrameRenderer(10)]:
    df = df.iloc[:100]
    return df


@task(interruptible=True, requests=Resources(cpu="1", mem="2Gi"))
def check_length(df: pd.DataFrame) -> str:
    return str(len(df))
