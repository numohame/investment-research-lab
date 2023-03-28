import json
import logging
from argparse import ArgumentParser
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from database import AWSClient
from portfolio_analytics.risk_metrics import (risk_metrics,
                                              risk_metrics_selector_config)

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
# list series which have been modeled as target
targets_path = "data/polygon_series.csv"
targets_file = pd.read_csv(targets_path)
targets_ids = list(targets_file["ticker"])

now = datetime.utcnow()
run_time = now.strftime("%Y%m%d%H%M%S")
ANNUALIZATION_FACTOR = 252

model_list = [
    # ENSEMBLE ESTIMATORS
    "Ada Boost Regressor",
    "Bagging Regressor",
    # "Extra Trees Regressor",  # slow
    "Gradient Boosting Regressor",
    # "Hist Gradient Boosting Regressor", # very slow
    "Random Forest Regressor",  # slow
    # LINEAR ESTIMATORS
    "ARD Regressor",
    "Bayesian Ridge Regressor",
    "Passive Aggressive Regressor",
    "Ridge Regressor",
    "SGD Regressor",
    # NEIGHBORS ESTIMATORS
    "K Neighbors Regressor",
    # SUPPORT VECTOR ESTIMATORS
    "Support Vector Regressor",
    "Nu SVR Regressor",
    # TREE ESTIMATORS
    "Decision Tree Regressor",
    "Extra Tree Regressor",
    # GAUSSIAN PROCESS ESTIMATORS
    "Gaussian Process Regressor",
]


if __name__ == "__main__":
    parser = ArgumentParser(description="")

    parser.add_argument(
        "--bucket",
        type=str,
        default="investment-research-lab",
        help="S3 bucket",
    )
    parser.add_argument(
        "--indicator_model",
        type=str,
        help="fundamental_data model",
        default="['fundamental']",
    )
    parser.add_argument(
        "--summary_type",
        type=str,
        help="backtesting",
        default="backtest",
    )
    parser.add_argument(
        "--num_best_features",
        type=str,
        help="number of best features desired",
        default="['3']",
    )
    parser.add_argument(
        "--quantile_list",
        type=str,
        default="[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]",
        help="enables assessment of stock performance distribution post trend change "
        "event",
    )
    parser.add_argument(
        "--fit_fraction",
        type=float,
        help="fraction of sample to use when fitting, pre-predict",
        default=1,
    )
    parser.add_argument(
        "--seed_fraction",
        type=float,
        help="fraction of sample to use as seed when predicting",
        default=0.4,
    )
    parser.add_argument(
        "--target_definition",
        type=str,
        help="variable targeted for prediction",
        default="['close']",
    )
    parser.add_argument(
        "--forecast_adj",
        type=str,
        help="adjust forecast based on model error",
        default="['False']",
    )
    parser.add_argument(
        "--backtest_horizon",
        type=str,
        help="length of model fit_predict sample at each backtest date",
        default="['20']",
    )
    parser.add_argument(
        "--num_backtest_periods",
        type=str,
        help="number of backtest periods",
        default="['40']",
    )
    parser.add_argument(
        "--universe_partition",
        type=float,
        help="fraction of universe of targets to predict [useful for testing purposes "
        "to limit universe of targets]",
        default=1,
    )
    parser.add_argument(
        "--forecast_horizon",
        type=str,
        help="length of forecast [in unit interval -- one quarter]",
        default="next_fin rprt date",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="confidence level when estimating tail risk",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        help="time at which structured dataset was created",
        default=str(["20230305145929"]),
    )
    parser.add_argument(
        "--secrets",
        type=str,
        help="path to secrets file json",
        default="investment-research-lab-secrets.json",
    )
    args = parser.parse_args()

    with open(args.secrets, "r") as f:
        secrets = json.load(f)

    aws_client = AWSClient(
        aws_access_key_id=secrets["aws_access_key_id"],
        aws_secret_access_key=secrets["aws_secret_access_key"],
        region_name=secrets["aws_region"],
    )

    # Below are target_ids which are variable put into the open file command
    targets_ids = targets_ids[: int(len(targets_ids) * args.universe_partition)]

    # open the timestamps command line argument
    timestamps = args.timestamp
    timestamps = timestamps.replace("'", '"')
    """Note: read-in list of timestamps via json"""
    timestamps = json.loads(timestamps)

    # open the indicator_model command line argument
    indicator_model = args.indicator_model
    indicator_model = indicator_model.replace("'", '"')
    """Note: read-in list of indicator_model via json"""
    indicator_model = json.loads(indicator_model)

    # open the quantile_list command line argument
    quantile_list = args.quantile_list
    quantile_list = quantile_list.replace("'", '"')
    """Note: read-in list of timestamps via json"""
    quantile_list = json.loads(quantile_list)

    # open the target_definition command line argument
    target_definition = args.target_definition
    target_definition = target_definition.replace("'", '"')
    """Note: read-in list of target_definition via json"""
    target_definition = json.loads(target_definition)

    # open the target_definition command line argument
    summary_type = args.summary_type

    # open the forecast_horizon command line argument
    forecast_horizon = args.forecast_horizon

    # open the forecast_adj command line argument
    forecast_adj = args.forecast_adj
    forecast_adj = forecast_adj.replace("'", '"')
    """Note: read-in list of forecast_adj via json"""
    forecast_adj = json.loads(forecast_adj)

    # open the num_best_features command line argument
    num_best_features = args.num_best_features
    num_best_features = num_best_features.replace("'", '"')
    """Note: read-in list of  num_best_features via json"""
    num_best_features = json.loads(num_best_features)

    # open the backtest_horizon command line argument
    backtest_horizon = args.backtest_horizon
    backtest_horizon = backtest_horizon.replace("'", '"')
    """Note: read-in list of backtest_horizon via json"""
    backtest_horizon = json.loads(backtest_horizon)

    # open the num_backtest_periods command line argument
    num_backtest_periods = args.num_backtest_periods
    num_backtest_periods = num_backtest_periods.replace("'", '"')
    """Note: read-in list of num_backtest_periods via json"""
    num_backtest_periods = json.loads(num_backtest_periods)

    """Note: create a 1-day and N-minute object from lag and resolution. Enables 
    needed forecast horizon granularity when opening classif_diagnostics files."""

    summary_results_list = []
    for target_id in targets_ids:
        backtester_results_params_list = []
        for a, b, c, d, e, f, g in product(
            indicator_model,
            num_best_features,
            backtest_horizon,
            num_backtest_periods,
            forecast_adj,
            target_definition,
            timestamps,
        ):
            params = {
                "indicator_model": a,
                "num_best_features": b,
                "backtest_horizon": c,
                "num_backtest_periods": d,
                "forecast_adj": e,
                "target_definition": f,
                "timestamp": g,
            }
        params["target_id"] = target_id
        backtester_results_params_list.append(params)
        # 1. iterate through each workbook in the given path, extract necessary data and
        # place into object
        for idx in range(0, len(backtester_results_params_list)):
            benchmark_backtest_periods_list = []
            parameters = backtester_results_params_list[idx]
            model = parameters["indicator_model"]
            for ix_model in range(0, len(model_list)):
                model_name = model_list[ix_model]
                filename = (
                    f"{model}_indicators/"
                    f"{summary_type}/results/"
                    f"{model_name}_tgt_pred_num_best_feat={parameters['num_best_features']}_"
                    f"fit_frac={args.fit_fraction}_"
                    f"seed_frac={args.seed_fraction}_"
                    f"btest_hzon={parameters['backtest_horizon']}_prds_"
                    f"num_btest_periods={parameters['num_backtest_periods']}_prds_"
                    f"fcast_hzon={forecast_horizon}_"
                    f"fcast_adj={parameters['forecast_adj'][0]}_"
                    f"tkr={parameters['target_id']}_"
                    f"{parameters['timestamp']}.csv"
                )

                # check if filename exists
                my_file = Path(filename)
                if my_file.is_file():
                    backtest_results_df = pd.read_csv(filename)
                    # remove the datetime localize and then filter out all records between
                    # 9:30:00 and 10:00:00 US/Eastern
                    backtest_results_df["forecast_date"] = pd.to_datetime(
                        backtest_results_df["forecast_date"], utc=True
                    )
                    backtest_results_df["forecast_date"] = backtest_results_df[
                        "forecast_date"
                    ].dt.tz_convert("US/Eastern")
                    backtest_results_df.set_index(["forecast_date"], inplace=True)
                    backtest_results_df = backtest_results_df.loc[
                        backtest_results_df.index.hour != 9
                    ]
                    annualization_factor = 4
                    # config the risk metrics class
                    risk_metrics_config = risk_metrics_selector_config(
                        backtest_binary_pred=backtest_results_df[["pred_binary_ret"]],
                        actual_ret=backtest_results_df[
                            [f"actual_ret_{forecast_horizon}"]
                        ],
                    )
                    risk_metrics_detail = risk_metrics(risk_metrics_config)
                    returns = risk_metrics_detail.backtester_returns()

                    # get the benchmark into the pnl dictionary at top
                    if num_backtest_periods not in benchmark_backtest_periods_list:
                        benchmark_returns = risk_metrics_detail.backtester_returns(
                            benchmark=True
                        )
                        total_ret = (1 + benchmark_returns).cumprod() - 1
                        if annualization_factor <= 4:
                            annual_ret = np.mean(benchmark_returns) * (
                                annualization_factor
                            )
                            annual_vol = np.std(benchmark_returns) * (
                                annualization_factor**0.5
                            )
                            sharpe_ratio = annual_ret / annual_vol
                        else:
                            annual_ret = np.nan
                            annual_vol = np.nan
                            vol = np.std(benchmark_returns) * (
                                len(backtest_results_df) ** 0.5
                            )
                            sharpe_ratio = total_ret[-1:] / vol
                        # calculate VaR, CVaR, max drawdown
                        rm_dict = risk_metrics_detail.risk_measurement(
                            benchmark_returns, args.alpha
                        )

                        # estimate benchmark PnL if not estimated at given num_btest_periods
                        summary_results_list.append(
                            {
                                "ticker": parameters["target_id"],
                                "model_indicator_class": "benchmark",
                                "estimator": model_name,
                                "forecast_adj": forecast_adj[0],
                                "forecast_horizon": forecast_horizon,
                                "backtest_horizon": parameters["backtest_horizon"],
                                "backtest_periods": parameters["num_backtest_periods"],
                                "num_best_features": parameters["num_best_features"],
                                "target_definition": parameters["target_definition"],
                                "annual_return": f"{annual_ret:.2%}",
                                "annual_volatility": f"{annual_vol:.2%}",
                                "total_ret": f"{total_ret[-1:][0]:.2%}",
                                "sharpe_ratio": f"{sharpe_ratio:.2}",
                                f"{100*(1-args.alpha):.0f}%_VaR": f"{rm_dict['VaR']:.2%}",
                                f"{100*(1-args.alpha):.0f}%_CVaR": f"{rm_dict['CVaR']:.2%}",
                                "max_drawdown": f"{rm_dict['mdd']:.2%}",
                                "average_drawdown": f"{rm_dict['add']:.2%}",
                                "worst_realization": f"{rm_dict['wr']:.2%}",
                            }
                        )
                        benchmark_backtest_periods_list.append(num_backtest_periods)

                    # estimate strategy PnL
                    total_ret = (1 + returns).cumprod() - 1
                    if annualization_factor <= 4:
                        annual_ret = np.mean(returns) * (annualization_factor)
                        annual_vol = np.std(returns) * (annualization_factor**0.5)
                        sharpe_ratio = annual_ret / annual_vol
                    else:
                        annual_ret = np.nan
                        annual_vol = np.nan
                        vol = np.std(returns) * (len(backtest_results_df) ** 0.5)
                        sharpe_ratio = total_ret[-1:] / vol
                    # calculate VaR, CVaR, max drawdown
                    rm_dict = risk_metrics_detail.risk_measurement(returns, args.alpha)

                    summary_results_list.append(
                        {
                            "ticker": parameters["target_id"],
                            "model_indicator_class": model,
                            "estimator": model_name,
                            "forecast_adj": forecast_adj[0],
                            "forecast_horizon": forecast_horizon,
                            "backtest_horizon": parameters["backtest_horizon"],
                            "backtest_periods": parameters["num_backtest_periods"],
                            "num_best_features": parameters["num_best_features"],
                            "target_definition": parameters["target_definition"],
                            "annual_return": f"{annual_ret:.2%}",
                            "annual_volatility": f"{annual_vol:.2%}",
                            "total_ret": f"{total_ret[-1:][0]:.2%}",
                            "sharpe_ratio": f"{sharpe_ratio:.2}",
                            f"{100*(1-args.alpha):.0f}%_VaR": f"{rm_dict['VaR']:.2%}",
                            f"{100*(1-args.alpha):.0f}%_CVaR": f"{rm_dict['CVaR']:.2%}",
                            "max_drawdown": f"{rm_dict['mdd']:.2%}",
                            "average_drawdown": f"{rm_dict['add']:.2%}",
                            "worst_realization": f"{rm_dict['wr']:.2%}",
                        }
                    )

        # convert list of dictionaries to pandas dataframe
        summary_results_df = pd.DataFrame(summary_results_list)

    # group by ticker
    # estimate each ML model's relative sharpe vs. the long-only buy-hold ["bmark"]
    summary_results_df["sharpe_ratio"] = summary_results_df["sharpe_ratio"].astype(
        float
    )
    summary_results_df_grouped = summary_results_df.groupby("ticker")
    relative_sharpe = []
    # relative_perf = []
    for name, group in summary_results_df_grouped:
        bmk_sharpe = group["sharpe_ratio"].iloc[0]
        for ix in range(0, len(group)):
            relative_sharpe.append(group["sharpe_ratio"].iloc[ix] - bmk_sharpe)
    summary_results_df["relative_sharpe_ratio"] = relative_sharpe
    # summary_results_df["relative_return"] = relative_perf
    cols = ["sharpe_ratio", "relative_sharpe_ratio"]
    summary_results_df[cols] = summary_results_df[cols].applymap(
        lambda x: "{" "0:.2f}".format(x)
    )
    LOGGER.info("group modeled")
    # set aws output path
    filepath_out = f"{model}_indicators/{args.summary_type}/summary_results/pnl"
    # load val to S3 folder
    aws_client.upload_object(
        bucket=args.bucket,
        filename=f"{filepath_out}/"
        f"summary_results_pnl_"
        f"fcast_hzon=next_fin rprt date_"
        f"{run_time}.csv",
        fileobj=summary_results_df.to_csv(index=True),
    )
    # save as csv on local
    summary_results_df.to_csv(
        f"{filepath_out}/"
        f"summary_results_pnl_"
        f"fcast_hzon=next_fin rprt date_"
        f"{run_time}.csv",
    )

    # create quantile distribution of sharpe ratio
    # answer questions like, are the highest sharpe ratios associated with:
    summary_results_df["relative_sharpe_ratio"] = summary_results_df[
        "relative_sharpe_ratio"
    ].astype(float)
    # 1) particular estimator models and stocks;
    summary_results_df_outperf_estimator = summary_results_df.loc[
        summary_results_df["relative_sharpe_ratio"] > 0
    ]
    summary_results_df_outperf_estimator_grouped = (
        summary_results_df_outperf_estimator.groupby("ticker")
    )
    summary_results_df_outperf_estimator = (
        summary_results_df_outperf_estimator_grouped.filter(lambda x: len(x) >= 6)
    )
    # save as csv on local
    summary_results_df_outperf_estimator.to_csv(
        f"{filepath_out}/"
        f"summary_results_pnl_outperf_estimator"
        f"fcast_hzon=next_fin rprt date_"
        f"{run_time}.csv",
    )
    LOGGER.info(
        "we have the summary results of relative risk-adjusted perf, "
        "decomposed by quantile with focus ont those stocks which outperform "
        "their individual benchmark across at least half of all ML Models"
    )
