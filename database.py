import functools
import logging
import os
from collections import OrderedDict
from datetime import datetime
from io import BytesIO
from time import sleep
from typing import IO, Any, Callable, Iterable, List, Mapping

import boto3
import numpy as np
import pandas as pd
from botocore.errorfactory import ClientError
from pyathena.connection import Connection
from pyathena.pandas.cursor import PandasCursor

LOGGER = logging.getLogger(__name__)

JAVA_TS_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
ATHENA_DATE_FORMAT = "%Y-%m-%d"


def retry_and_log(retry_attempts: int = 10) -> Callable[..., Any]:
    """
    Retry a function, logging exceptions. Ignores the output of the function.
    """

    def wrapper(f: Callable[..., Any]) -> Callable[..., bool]:
        @functools.wraps(f)
        def inner(*args: Any, **kwargs: Any) -> bool:
            for i in range(retry_attempts):
                try:
                    f(*args, **kwargs)
                    return True
                except Exception as err:
                    sleep(i + 1)
                    LOGGER.info(f"Exception: {err}")
            return False

        return inner

    return wrapper


def convert_to_athena_ts_format(timestamps: Iterable) -> Any:
    return pd.to_datetime(timestamps).dt.strftime(JAVA_TS_FORMAT)


def convert_to_athena_date_format(dates: Iterable) -> Any:
    return pd.to_datetime(dates).dt.strftime(ATHENA_DATE_FORMAT)


# TODO: Remove self.bucket_name since AWS clients aren't connected to buckets.
# Better to have as function params.
class AWSClient:
    """
    Client to interact with AWS
    """

    def __init__(
        self,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        region_name: str = "us-east-2",
        bucket_name: str = "investment-research-lab",
        athena_staging: str = "s3://investment-research-lab/athena-results/",
    ) -> None:
        """
        connect to the AWS DBs
        """
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.region_name = region_name
        self.bucket_name = bucket_name
        self.athena_staging = athena_staging
        # Resource client
        self._s3_resource = boto3.resource(
            "s3",
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
        # S3 client
        self._s3_client = boto3.client(
            "s3",
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
        # Pandas cursor
        self._cursor = Connection(
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            s3_staging_dir=self.athena_staging,
            region_name=self.region_name,
            cursor_class=PandasCursor,
        ).cursor()

    def query_athena(self, query: str) -> pd.DataFrame:
        data = self._cursor.execute(query).as_pandas()
        LOGGER.debug(f"query:\n{query}\n============")
        return data

    def file_exists(self, path: str) -> bool:
        try:
            self._s3_client.head_object(Bucket=self.bucket_name, Key=path)
            return True
        except ClientError:
            return False

    def ls_s3(
        self,
        bucket: str,
        prefix: str = "",
        suffix: str = "",
        omit_folders: bool = True,
        as_url: bool = False,
    ) -> List[str]:
        """
        Lists files in s3://{bucket}/{prefix}{suffix}.
        Uses s3 resource to simplify code. Note that prefix MUST endin '/'
        """
        LOGGER.info(f"Reading objects from s3://{bucket}/{prefix}*{suffix}")

        files = []
        for file in self._ls(bucket, prefix):
            if file.key.endswith(suffix):
                files.append(file.key)
            elif omit_folders and (file.key[-1] != "/"):
                files.append(file.key)
        if as_url:
            files = [os.path.join("s3://", bucket, f) for f in files]

        return files

    def _ls(self, bucket: str, prefix: str) -> Any:
        return self._s3_resource.Bucket(bucket).objects.filter(
            Delimiter="/", Prefix=prefix
        )

    def download_csv_as_df(self, bucket: str, filename: str) -> pd.DataFrame:
        resp_bytes = self._get_object_contents(bucket, filename)
        return pd.read_csv(BytesIO(resp_bytes), low_memory=False)

    def download_s3_to_fileobj(self, bucket: str, prefix: str, file: IO) -> bool:
        LOGGER.info(f"Downloading s3://{bucket}/{prefix} to local file.")
        self._s3_client.download_fileobj(Bucket=bucket, Key=prefix, Fileobj=file)
        file.seek(0)
        return True

    def _get_object_contents(self, bucket: str, filename: str) -> bytes:
        resp = self._s3_client.get_object(Bucket=bucket, Key=filename)
        resp_bytes: bytes = resp["Body"].read()
        return resp_bytes

    def download_object(self, bucket: str, filename: str) -> bytes:
        return self._get_object_contents(bucket, filename)

    def download_s3_to_file(self, bucket: str, prefix: str, path: str) -> bool:
        LOGGER.info(f"Downloading s3://{bucket}/{prefix} to local file.")
        self._s3_client.download_file(Bucket=bucket, Key=prefix, Filename=path)
        return True

    def upload_object(self, bucket: str, filename: str, fileobj: str) -> dict:
        resp: dict = self._s3_client.put_object(
            Bucket=bucket, Key=filename, Body=fileobj
        )
        return resp

    def read_s3_parquet(self, path: str) -> pd.DataFrame:
        buffer = BytesIO()
        self._s3_client.download_fileobj(
            Bucket=self.bucket_name, Key=path, Fileobj=buffer
        )
        LOGGER.debug(f"reading parquet file from {path}")
        data = pd.read_parquet(buffer)
        return data

    def save_s3_parquet(
        self,
        data: pd.DataFrame,
        path: str,
        overwrite: bool = True,
        chunk_size_bytes: int = 500 * 10**6,
    ) -> bool:
        chunks = self._get_chunks(data, chunk_size_bytes)
        for n, df in enumerate(chunks):
            if len(chunks) == 1:
                path_part = path
            else:
                path_part = path.split(".")[0] + f"_part_{n}.parquet"

            if not overwrite:
                if self.file_exists(path):
                    LOGGER.info(f"File already exists at {path_part}, not overwriting.")
                    return False

            buffer = BytesIO()
            df.to_parquet(buffer, engine="pyarrow", compression=None, index=False)
            LOGGER.info(f"Saving to {path_part}")
            self._s3_client.put_object(
                Bucket=self.bucket_name,
                Key=path,
                Body=buffer.getvalue(),
            )

        return True

    def _get_chunks(
        self, df: pd.DataFrame, chunk_size_bytes: int = 500 * 10**6
    ) -> List[pd.DataFrame]:
        mem = df.memory_usage(index=True, deep=True).sum()
        n_chunks = int(mem / chunk_size_bytes)

        if n_chunks == 0:
            n_chunks = 1

        chunks: List[pd.DataFrame] = np.array_split(df, n_chunks)
        return chunks

    def save_file(self, local_path: str, path: str, overwrite: bool = True) -> bool:
        if not overwrite:
            if self.file_exists(path):
                LOGGER.debug(f"file already exists at {path}, will no overwrite.")
                return False
        self._s3_client.upload_file(
            Filename=local_path,
            Bucket=self.bucket_name,
            Key=path,
        )
        LOGGER.debug(f"saved file to {path}")
        return True

    def get_parquet_path(self, partition_keys: Mapping, data_dir: str) -> str:
        """
        get the path of the parquet file in the distributed file store:
        (partition_keys should probably be an OrderedDict)

            s3://bucket_name/data_dir/key=val/.../data.parquet
        """
        if data_dir[-1] == "/":
            path = data_dir
        else:
            path = data_dir + "/"
        for key, val in partition_keys.items():
            if isinstance(val, datetime):
                val = val.strftime("%Y-%m-%d")
            path += f"{key}={val}/"
        path = path + "data.parquet"
        return path

    def get_parquet_path_no_partition(self, data_dir: str) -> str:
        """
        get the path of the parquet file in the distributed file store:
        (partition_keys should probably be an OrderedDict)

            s3://bucket_name/data_dir/key=val/.../data.parquet
        """
        if data_dir[-1] == "/":
            path = data_dir
        else:
            path = data_dir + "/"
        # for key, val in partition_keys.items():
        #     if isinstance(val, datetime):
        #         val = val.strftime("%Y-%m-%d")
        #     path += f"{key}={val}/"
        path = path + "data.parquet"
        return path

    def append_to_dataset(
        self,
        data: pd.DataFrame,
        partition_cols: List[str],
        data_dir: str,
    ) -> None:
        """
        append new data to an existing dataset in S3
        """
        for keys, df in data.groupby(partition_cols):
            partition_keys = pd.Series(keys, partition_cols).to_dict(OrderedDict)
            path = self.get_parquet_path(partition_keys, data_dir)
            if self.file_exists(path):
                existing_data = self.read_s3_parquet(path)
                new_data = pd.concat([existing_data, df], ignore_index=True)
            else:
                new_data = df
            assert self.save_s3_parquet(new_data, path, overwrite=True)

    def merge_into_dataset(
        self,
        data: pd.DataFrame,
        partition_cols: List[str],
        key_cols: List[str],
        data_dir: str,
        resolution: str = None,
        indicator_window: str = None,
        alert_date: bool = None,
        polygon_slice: bool = None,
    ) -> None:
        """
        merge new data into an existing dataset in S3
        """
        for keys, df in data.groupby(partition_cols):
            partition_keys = pd.Series(keys, partition_cols).to_dict(OrderedDict)
            path = self.get_parquet_path(partition_keys, data_dir)
            if indicator_window:
                path = str.split(path, "/")
                path = (
                    f"{path[0]}/freq_{resolution}_window_{indicator_window}/"
                    f"{path[1]}/{path[2]}"
                )
            elif resolution:
                path = str.split(path, "/")
                path = f"{path[0]}/freq_{resolution}/{path[1]}/{path[2]}"
            if self.file_exists(path) and indicator_window:
                # keep only rows where index not in existing data
                existing_data = self.read_s3_parquet(path)
                df.rename(columns={"series_id": "id"}, inplace=True)
                new_data = pd.concat([existing_data, df])
                new_data = new_data.groupby(["date"]).first()
                new_data = new_data.reset_index()
            elif self.file_exists(path) and alert_date:
                # keep only rows where index not in existing data
                existing_data = self.read_s3_parquet(path)
                df.rename(columns={"series_id": "id"}, inplace=True)
                if alert_date:
                    df.rename(columns={"date": "alert_date"}, inplace=True)
                new_data = pd.concat([existing_data, df])
                new_data = new_data.groupby(key_cols).last()
                new_data = new_data.reset_index()
            elif self.file_exists(path):
                existing_data = self.read_s3_parquet(path)
                df.rename(columns={"series_id": "id"}, inplace=True)
                if polygon_slice:
                    df.rename(columns={"ticker": "ticker_id"}, inplace=True)
                new_data = pd.concat([existing_data, df])
                new_data = new_data.groupby(["date"]).last()
                new_data = new_data.reset_index()
            else:
                df.rename(columns={"series_id": "id"}, inplace=True)
                if alert_date:
                    df.rename(columns={"date": "alert_date"}, inplace=True)
                if polygon_slice:
                    df.rename(columns={"ticker": "ticker_id"}, inplace=True)
                new_data = df.copy()
            assert self.save_s3_parquet(new_data, path, overwrite=True)

    def merge_into_dataset_no_partition(
        self,
        data: pd.DataFrame,
        key_cols: List[str],
        data_dir: str,
    ) -> None:
        """
        merge new data into an existing dataset in S3
        """
        path = self.get_parquet_path_no_partition(data_dir)
        if self.file_exists(path):
            # keep only rows where index not in existing data
            existing_data = self.read_s3_parquet(path)
            new_data = pd.concat([existing_data, data])
            new_data = new_data.groupby(key_cols).last()
            new_data = new_data.reset_index()
        else:
            data.rename(columns={"series_id": "id"}, inplace=True)
            new_data = data.copy()
        assert self.save_s3_parquet(new_data, path, overwrite=True)

        return data
