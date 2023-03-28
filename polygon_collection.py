import json
import logging
from argparse import ArgumentParser
from datetime import datetime, timedelta

import pandas as pd

from apis.polygon import POLYGON
from database import AWSClient, retry_and_log

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

now = datetime.utcnow()
end_date = now.strftime("%Y%m%d%H%M%S")

SERIES_FILE = "data/polygon_series.csv"
series = pd.read_csv(SERIES_FILE, index_col=0)
series_ids = list(series.index)

# dict of multiplier and timespan for iteration across 1m and 1d

resolution_dict = {
    "1d": {"multiplier": 1, "timespan": "day"},
}

parser = ArgumentParser(description="")
parser.add_argument(
    "--minutes_lookback",
    type=int,
    help="length of historical inputs [unit = day]",
    default=10e3,
)
parser.add_argument(
    "--minute_data_looping",
    type=int,
    help="number of non-overlapping windows over which to gather minute freq data",
    default=1,
)
parser.add_argument(
    "--days_lookback",
    type=int,
    help="length of historical inputs [unit = day]",
    default=9000,
)
parser.add_argument(
    "--secrets",
    type=str,
    help="path to secrets file json",
    default="investment-research-lab-secrets.json",
)

args = parser.parse_args()

# polygon credentials
with open(args.secrets, "r") as f:
    secrets = json.load(f)
polygon_api = POLYGON(secrets["polygon_api_key"])


@retry_and_log(retry_attempts=1)
def pull_series(
    series_id: str,
    polygon_api: POLYGON,
    aws_client: AWSClient,
    start_date: str,
    end_date: str,
    multiplier: int,
    timespan: str,
) -> None:
    data = polygon_api.pull_observations(
        series_id, start_date, end_date, multiplier, timespan
    )

    data = data[
        [
            "date",
            "series_id",
            "name",
            "industry",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "vwap",
            "transactions",
        ]
    ]
    data["volume"] = data["volume"].astype(float)
    data["transactions"] = data["transactions"].astype(float)

    # keep the trading hour data [in UTC]
    if timespan == "minute":
        data = data.set_index("date").between_time("13:30", "20:00").reset_index()
    aws_client.merge_into_dataset(
        data=data,
        partition_cols=["series_id"],
        key_cols=[
            "date",
            "name",
            "industry",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "vwap",
            "transactions",
        ],
        data_dir="polygon_data/",
        resolution=f"{timespan}",
    )


if __name__ == "__main__":
    aws_client = AWSClient(
        aws_access_key_id=secrets["aws_access_key_id"],
        aws_secret_access_key=secrets["aws_secret_access_key"],
        region_name=secrets["aws_region"],
    )

    for key, val in resolution_dict.items():
        multiplier = val["multiplier"]
        timespan = val["timespan"]
        if timespan == "day":
            days_to_subtract = int(args.days_lookback)
            start_date = (
                pd.to_datetime(end_date) - timedelta(days=days_to_subtract)
            ).strftime("%Y-%m-%d")
            for id in range(len(series_ids)):
                series_id = series.index[id]
                LOGGER.info(f"pulling daily observations for {series_id}")
                data = pull_series(
                    series_id,
                    polygon_api,
                    aws_client,
                    start_date,
                    end_date,
                    multiplier,
                    timespan,
                )
        else:
            for idx in range(0, args.minute_data_looping):
                days_to_subtract = int(args.minutes_lookback / 365)
                start_date = (
                    pd.to_datetime(end_date) - timedelta(days=days_to_subtract)
                ).strftime("%Y-%m-%d")
                for id in range(len(series_ids)):
                    series_id = series.index[id]
                    LOGGER.info(
                        f"pulling observations for {series_id} in " f"loop number {idx}"
                    )
                    data = pull_series(
                        series_id,
                        polygon_api,
                        aws_client,
                        start_date,
                        end_date,
                        multiplier,
                        timespan,
                    )
                end_date = start_date
