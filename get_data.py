import json
from argparse import ArgumentParser
from datetime import datetime
from typing import List

import pandas as pd

from database import AWSClient

now = datetime.utcnow()
timestamp = now.strftime("%Y%m%d%H%M%S")

# list series to be transformed
targets_path = "data/polygon_series.csv"
targets_file = pd.read_csv(targets_path)
targets_ids = list(targets_file["ticker"])


def structured_data_to_AWS(
    df: pd.DataFrame,
    aws_client: str,
    bucket_out: str,
    filepath_out_aws: str,
    timestamp: str,
) -> None:
    aws_client.upload_object(
        bucket=bucket_out,
        filename=f"{filepath_out_aws}_" f"{timestamp}.csv",
        fileobj=df.to_csv(index=True),
    )
    print(f"structured data saved to url = s3://{filepath_out_aws}_" f"{timestamp}")


def pull_aws_polygon(
    aws_client: AWSClient,
    targets_ids: str,
    min_date: str,
) -> List[pd.DataFrame]:
    query = f"""
            SELECT
            date,
            series_id as id, 
            name as description,
            open,
            high,
            low,
            close,
            volume       
            FROM market_prices.polygon_data
            WHERE series_id in ('{targets_ids}') and
            date > CAST('{min_date}' as date)
            ORDER BY series_id, date
            """
    data = aws_client.query_athena(query)

    return data


def pull_aws_fundamenatal_data(
    aws_client: AWSClient,
) -> List[pd.DataFrame]:
    query = f"""
            SELECT
            *          
            FROM fundamental_indicators.fundamental_data
            ORDER BY ticker,reportperiod
            """
    data = aws_client.query_athena(query)

    return data


if __name__ == "__main__":
    parser = ArgumentParser(description="")

    parser.add_argument(
        "--indicator_model",
        type=str,
        help="fundamental_data model",
        default="fundamental",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        help="frequency of data -- unit [quarterly]",
        default="day",
    )
    parser.add_argument(
        "--universe_partition",
        type=str,
        help="fraction of universe of targets to predict [useful for testing "
        "purposes to limit universe of targets]",
        default="[0, 1]",
    )
    parser.add_argument(
        "--period_lookback",
        type=int,
        help="length of historical inputs",
        default=60,
    )
    parser.add_argument(
        "--update_features",
        type=bool,
        help="are we updating the feature set?",
        default=True,
    )
    parser.add_argument(
        "--bucket_out",
        type=str,
        help="S3 bucket to save results csv",
        default="investment-research-lab",
    )
    parser.add_argument(
        "--secrets",
        type=str,
        help="path to secrets file json",
        default="investment-research-lab-secrets.json",
    )

    args = parser.parse_args()

    # open the universe_partition command line argument
    universe_partition = args.universe_partition
    universe_partition = universe_partition.replace("'", '"')
    """Note: read-in list of num_best_features via json"""
    universe_partition = json.loads(universe_partition)

    # set aws output path
    filepath_out_aws = (
        f"{args.indicator_model}_indicators/features/" f"{args.indicator_model}_factors"
    )

    # set local output path
    filepath_out_local = (
        f"{args.indicator_model}_indicators/features/" f"{args.indicator_model}_factors"
    )

    # AWS credentials
    with open(args.secrets, "r") as f:
        secrets = json.load(f)

    aws_client = AWSClient(
        aws_access_key_id=secrets["aws_access_key_id"],
        aws_secret_access_key=secrets["aws_secret_access_key"],
        region_name=secrets["aws_region"],
    )

    # ---------------------GET DATA

    # 1. import data

    # partition the universe of targets to smaller list, if needed
    targets_ids_list = targets_ids[
        int(universe_partition[0] * len(targets_ids)) : int(
            universe_partition[1] * len(targets_ids)
        )
    ]
    targets_ids = "','".join(list(targets_ids_list))

    # 1.a. get the fundamental data and the stock price data
    # structure the fundamental data and stock price data

    df_fundamental_data = pull_aws_fundamenatal_data(aws_client)
    df_fundamental_data.rename(
        columns={"ticker": "id", "datekey": "date"}, inplace=True
    )
    df_fundamental_data["date"] = pd.to_datetime(
        df_fundamental_data["date"]
    ).dt.strftime("%Y-%m-%d")
    df_fundamental_data = df_fundamental_data.sort_values(
        ["id", "date"], ascending=[True, True]
    )
    start_date = pd.to_datetime(df_fundamental_data["date"]).min().strftime("%Y-%m-%d")
    df_polygon = pull_aws_polygon(aws_client, targets_ids, start_date)
    df_polygon["date"] = df_polygon["date"].dt.strftime("%Y-%m-%d")
    df = pd.merge(df_polygon, df_fundamental_data, on=["id", "date"], how="outer")
    df_grouped = df.groupby(["id"])
    group_list = []
    for name, group in df_grouped:
        group = group.dropna(subset=["name"])
        group_list.append(group)
    df_filtered = pd.concat(group_list)

    # 2. put the structured data to AWS bucket with timestamp
    structured_data_to_AWS(
        df_filtered,
        aws_client,
        args.bucket_out,
        filepath_out_aws,
        timestamp,
    )
