import json
import logging
from argparse import ArgumentParser
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import sklearn.metrics as validation_metrics
from scipy.stats import percentileofscore
from sklearn import preprocessing
from sklearn.ensemble import (AdaBoostRegressor, BaggingRegressor,
                              ExtraTreesRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.feature_selection import SequentialFeatureSelector as sfs
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import (ARDRegression, BayesianRidge,
                                  PassiveAggressiveRegressor, Ridge,
                                  SGDRegressor)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, NuSVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor

from database import AWSClient
from distributions import StudentT

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# list series to be modeled as target
targets_path = "c://users/investment-research-lab/data/polygon_series.csv"
targets_file = pd.read_csv(targets_path)
targets_ids = list(targets_file["ticker"])

model_list = [
    {
        # ENSEMBLE ESTIMATORS
        "ensemble": {
            "Ada Boost Regressor": AdaBoostRegressor(),
            "Bagging Regressor": BaggingRegressor(),
            # {"Extra Trees Regressor": ExtraTreesRegressor()},  # slow
            "Gradient Boosting Regressor": GradientBoostingRegressor(),
            # {"Hist Gradient Boosting Regressor": HistGradientBoostingRegressor()}, # very slow
            "Random Forest": RandomForestRegressor(),  # slow
        },
        # ENSEMBLE ESTIMATORS
        "ensemble_ada_boost": {"Ada Boost Regressor": AdaBoostRegressor()},
        "ensemble_bagging": {"Bagging Regressor": BaggingRegressor()},
        # {"Extra Trees Regressor": ExtraTreesRegressor()},  # slow
        "ensemble_gradient_boost": {
            "Gradient Boosting Regressor": GradientBoostingRegressor(),
        },
        # {"Hist Gradient Boosting Regressor": HistGradientBoostingRegressor()},  # very slow
        "ensemble_random_forest": {
            "Random Forest": RandomForestRegressor(),
        },  # slow
        "non-ensemble": {
            # LINEAR ESTIMATORS
            "ARD Regressor": ARDRegression(),
            "Bayesian Ridge Regressor": BayesianRidge(),
            "Passive Aggressive Regressor": PassiveAggressiveRegressor(),
            "Ridge Regressor": Ridge(),
            "SGD Regressor": SGDRegressor(),
            # NEIGHBORS ESTIMATORS
            "K Neighbors Regressor": KNeighborsRegressor(),
            # SUPPORT VECTOR ESTIMATORS
            "Support Vector Regressor": SVR(),
            "Nu SVR": NuSVR(),
            # TREE ESTIMATORS
            "Decision Tree Regressor": DecisionTreeRegressor(),
            "Extra Tree Regressor": ExtraTreeRegressor(),
            # GAUSSIAN PROCESS ESTIMATORS
            "Gaussian Process Regressor": GaussianProcessRegressor(),
        },
    }
]

"""Note: generally good to normalize prior to fit/predict. This reduces scale
differences between variables ."""
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

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
        default="fundamental",
    )
    parser.add_argument(
        "--estimator_model_class",
        type=str,
        help="ensemble, ensemble_{model_name}, or non-ensemble",
        default="non-ensemble",
    )
    parser.add_argument(
        "--num_best_features",
        type=int,
        help="number of best features desired",
        default=3,
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
        "--target",
        type=str,
        help="specify your target variable",
        default="close",
    )
    parser.add_argument(
        "--backtest_horizon",
        type=int,
        help="length of model fit_predict sample at each backtest date",
        default=20,
    )
    parser.add_argument(
        "--num_backtest_periods",
        type=int,
        help="number of backtest periods",
        default=40,
    )
    parser.add_argument(
        "--forecast_horizon",
        type=int,
        help="length of forecast [in unit interval -- one quarter]",
        default=1,
    )
    parser.add_argument(
        "--universe_partition",
        type=str,
        help="fraction of universe of targets to predict [useful for testing purposes "
        "to limit universe of targets]",
        default="[0, 1]",
    )
    parser.add_argument(
        "--forecast_adj",
        type=str,
        help="adjust forecast based on model error",
        default="False",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        help="time at which structured dataset was created",
        default="20230305145929",
    )
    parser.add_argument(
        "--secrets",
        type=str,
        help="path to secrets file json",
        default="c:/users/investment-research-lab-secrets.json",
    )

    args = parser.parse_args()

    with open(args.secrets, "r") as f:
        secrets = json.load(f)

    aws_client = AWSClient(
        aws_access_key_id=secrets["aws_access_key_id"],
        aws_secret_access_key=secrets["aws_secret_access_key"],
        region_name=secrets["aws_region"],
    )

    # set aws output path
    filepath_out_aws = f"{args.indicator_model}_indicators/backtest"

    # set local output path
    filepath_out_local = (
        f"c://users/investment-research-lab/"
        f"{args.indicator_model}_indicators/backtest"
    )

    # open the universe_partition command line argument
    universe_partition = args.universe_partition
    universe_partition = universe_partition.replace("'", '"')
    """Note: read-in list of num_best_features via json"""
    universe_partition = json.loads(universe_partition)

    # open the estimator model class command line argument
    estimator_model_class = args.estimator_model_class

    # open the num_best_features command line argument
    num_best_features = args.num_best_features

    # open the num_backtest_periods command line argument
    num_backtest_periods = args.num_backtest_periods

    # open the backtest_horizon command line argument
    backtest_horizon = args.backtest_horizon

    # 1) read in cleaned df with features
    # 1. import data
    targets_ids = targets_ids[
        int(universe_partition[0] * len(targets_ids)) : int(
            universe_partition[1] * len(targets_ids)
        )
    ]
    df = aws_client.download_csv_as_df(
        args.bucket,
        f"{args.indicator_model}_indicators/features/"
        f"{args.indicator_model}_factors_"
        f"{args.timestamp}.csv",
    )
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df["date"] = pd.to_datetime(df["date"])
    df.set_index(["date"], inplace=True)

    # 2) define the range of end_dates with associated start dates
    """Note: convert feature and target data to return space.
    Input: common frequency of target and feature data.
    Output = lag: forecast_horizon """
    lag = args.forecast_horizon
    TARGET = args.target
    FEATURES = df.columns[
        ~df.columns.isin(
            [
                "id",
                "description",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "name",
                "sector",
                "industry",
                "dimension",
                "sp500",
                "calendardate",
                "reportperiod",
                "lastupdated",
                "fxusd",
                "sharefactor",
                "stocknumber",
                "timeindex",
                "calendaryear",
                "marketcap",
                "price",
            ]
        )
    ]

    # make all FEATURE columns of type float

    # slice by list of targets in universe partition
    df = df.loc[df["id"].isin(targets_ids)]

    # groupby target_id
    df_grouped = df.groupby(["id"], as_index=False)
    for name, df in df_grouped:
        if df[TARGET].count() < (num_backtest_periods + backtest_horizon):
            num_backtest_periods = backtest_horizon = int(
                0.5 * (df[TARGET].count() - (lag + 1))
            )
        df[FEATURES] = df[FEATURES].astype(float)
        df.sort_index(inplace=True)
        y_df = df[TARGET][lag:]
        y_df = y_df[len(y_df) - num_backtest_periods - backtest_horizon - (lag + 1) :]
        y_df = y_df.dropna()
        if len(y_df) < args.backtest_horizon + lag + 1:
            LOGGER.info(
                f"the ticker {name} does not have sufficient length of closing "
                f"prices available at each relevant point in time within "
                f"the backtest"
            )
            continue
        y = df[TARGET][lag:] / df[TARGET][lag:].shift(lag) - 1
        y.dropna(how="any", inplace=True)
        y = np.array(y)
        # don't estimate change of FEATURES if contains substring "pctchg"
        # split FEATURES into FEATURES_nonpctchg and FEATURES_pctchg
        pctchg_cols = [col for col in df.columns if "pctchg" in col]
        X_pctchg = df[pctchg_cols].astype(float)
        FEATURES = [col for col in FEATURES if not "pctchg" in col]
        X_non_pctchg = (
            df[FEATURES].astype(float) / df[FEATURES].shift(lag).astype(float) - 1
        )
        X = pd.concat([X_non_pctchg, X_pctchg], axis=1)
        """Note: The important correlation between features and target is the predicive 
        correlation. This predictive correlation can be measured via lagging the 
        predictive features by the number of time steps we are predicting target.

        For example, if predicting target one step ahead, then lag the predictive 
        features by one time step."""

        # split the features object down to the backtest period
        # there are generally more nan in features in the first few periods,
        # therefore, reducing the features space to the backtest period will make
        # more likely to exclude the earliest periods
        # this results in fewer features being dropped, given a feature is dropped if
        # any nan in the feature's time series
        X = X[len(X) - num_backtest_periods - backtest_horizon - (lag + 1) :]
        X = X.shift(lag)
        X.replace([-np.inf, np.inf], np.nan, inplace=True)
        X.dropna(axis=0, how="all", inplace=True)
        X.dropna(axis=1, how="any", inplace=True)
        FEATURES = X.columns
        if len(FEATURES) <= num_best_features:
            continue
        X_arr = np.array(X)

        # backtest_end_dates = len(X)-args.backtest_horizon
        backtest_end_dates = df.index[-num_backtest_periods:]

        # 3) read in best features model-by-model
        multimodel_prediction_list = []  # multi-model:includes central forecast +
        # prob dist
        for ix_model in range(0, len(model_list[0][estimator_model_class])):
            target_prediction_list = []  # for central forecast per model
            distributional_levels_list = []  # for resampling
            selector_best_features = []  # for each estimator create empty list
            # for backtest_date in range(0, len(backtest_end_dates) - lag):
            for backtest_date in range(-len(backtest_end_dates), 0):
                start = timer()
                model_name = list(model_list[0][args.estimator_model_class].keys())[
                    ix_model
                ]
                reg_model = list(model_list[0][args.estimator_model_class].values())[
                    ix_model
                ]  # reg model used in iteration
                y_scaled = scaler.fit_transform(
                    y[
                        len(y)
                        + backtest_date
                        - backtest_horizon : len(y)
                        + backtest_date,
                        None,
                    ]
                )  # min-max scaled
                X_scaled = scaler.fit_transform(
                    X_arr[
                        len(X_arr)
                        + backtest_date
                        - backtest_horizon : len(X_arr)
                        + backtest_date
                    ]
                )  # min-max scaled
                selector = sfs(
                    reg_model,
                    n_features_to_select=num_best_features,
                )
                selector = selector.fit(X_scaled, y_scaled[:, 0])
                end = timer()
                LOGGER.info(
                    f"finding {num_best_features} best features takes "
                    f"{end-start} seconds "
                    f"with model {model_name}"
                )
                selector.best_features = selector.get_feature_names_out(FEATURES)
                selector_best_features.append(selector.best_features[None, :])
                best_features = pd.DataFrame(selector.best_features[None, :])

                # 3.a. slice the clean_features_df across only best model features
                best_features = best_features.iloc[:].values.tolist()
                best_features = [item for items in best_features for item in items]

                # 3.b. put predictors to X object
                seed = X[best_features][
                    len(X) + backtest_date - backtest_horizon : len(X) + backtest_date
                ]
                # seed = (
                #     df[best_features][
                #         len(df)
                #         + backtest_date
                #         - backtest_horizon : len(df)
                #         + backtest_date
                #     ]
                #     / df[best_features][
                #         len(df)
                #         + backtest_date
                #         - backtest_horizon : len(df)
                #         + backtest_date
                #     ].shift(lag)
                #     - 1
                # )
                seed = seed.iloc[-1 - int(args.seed_fraction * len(seed)) :, :]
                seed.dropna(how="any", inplace=True)
                seed_scaled = scaler.fit_transform(np.array(seed))
                X_best = X[best_features][len(X) + backtest_date - backtest_horizon :]
                # X_best = (
                #     X[best_features][
                #         len(X) + backtest_date - backtest_horizon - lag - 1:
                #     ]
                # )
                # X_best = (
                #     df[best_features][
                #         len(df) + backtest_date - backtest_horizon - lag - 1:
                #     ]
                #     / df[best_features][
                #       len(df) + backtest_date - backtest_horizon - lag - 1:
                #       ].shift(lag)
                #     - 1
                # )
                # X_best = X_best.shift(lag)
                X_best.dropna(how="any", inplace=True)
                X_best = np.array(X_best)
                X_best_scaled = scaler.fit_transform(
                    X_best[
                        len(X_best)
                        + backtest_date
                        - backtest_horizon : len(X_best)
                        + backtest_date
                    ]
                )
                y_scaled = scaler.fit_transform(
                    y[
                        len(y)
                        + backtest_date
                        - backtest_horizon : len(y)
                        + backtest_date,
                        None,
                    ]
                )  # min-max scaled
                reg_model.fit(
                    X_best_scaled[: int(args.fit_fraction * backtest_horizon)],
                    y_scaled[: int(args.fit_fraction * backtest_horizon)].ravel(),
                )

                target_predict = reg_model.predict(seed_scaled)
                end = timer()
                LOGGER.info(
                    f"running fit and predict with {num_best_features} best "
                    f"features "
                    f"when estimator model is {model_name} and backtester forecast "
                    f"date "
                    f"is {seed.index[-1].strftime('%Y-%m-%d')} and target_id "
                    f"is {name} takes {end - start} seconds"
                )
                target_predict = scaler.inverse_transform(target_predict[:, None])

                target_prediction_list.append(
                    {
                        model_name: {
                            "forecast_date": seed.index[-1].strftime("%Y-%m-%d"),
                            "target_name": TARGET,
                            "actual_closing_level": df[TARGET].iloc[
                                backtest_date - lag
                            ],
                            f"actual_level_next_fin rprt date": df[TARGET].iloc[
                                backtest_date
                            ],
                            f"actual_ret_next_fin rprt date": (
                                df[TARGET].iloc[backtest_date]
                                / df[TARGET][backtest_date - lag]
                            )
                            - 1,
                            f"pred_ret_next_fin rprt date": target_predict[0][0],
                            f"pred_level_next_fin rprt date": df[TARGET].iloc[
                                backtest_date - lag
                            ]
                            * (1 + target_predict[0][0]),
                            f"pred_error": (
                                target_predict[0][0]
                                - (
                                    df[TARGET].iloc[backtest_date]
                                    / df[TARGET].iloc[backtest_date - lag]
                                    - 1
                                )
                            ),
                        }
                    }
                )

            # 4. resample error distribution around the central forecast
            """Note: need to resample. Get error from pred_v_real.Open pred_v_real 
            consistent with estimator model,forecast_horizon, lookback."""

            # 4.a convert list of dict into dataframe
            target_prediction_df = pd.concat(
                [
                    pd.DataFrame.from_dict(x, orient="index")
                    for x in target_prediction_list
                ]
            )
            # 4.b resample model error for backtest dates past warm-up period
            # warm-up period defined as first backtest dates until # backtest dates
            # equal length of some arbitrary seed period
            seed_error = int(args.seed_fraction * len(target_prediction_df))
            for idx in range(seed_error, len(target_prediction_df)):
                error = target_prediction_df["pred_error"][idx - seed_error : idx]
                loc_eps = 0
                dof_eps = 5
                scale_eps = np.std(error)
                t_dist_eps = StudentT(loc_eps, scale_eps, dof_eps)  # t-dist errors
                results_error = t_dist_eps.sample()
                # resample error to distribute around model estimator predicted ret
                # let's temporarily test using  a central forecast adjusted by CV error
                if args.forecast_adj:
                    central_forecast = target_predict[0][0] - np.mean(error)
                else:
                    central_forecast = target_predict[0][0]
                central_forecast_arr = np.repeat(
                    np.array(central_forecast), len(results_error)
                )
                results = central_forecast_arr + results_error

                # 4c. estimate percentile distribution
                prob_below_zero = percentileofscore(results, 0)
                # compute percentiles of distribution
                percentiles = np.percentile(results, [2.5, 16, 25, 50, 75, 84, 97.5])
                # convert from percentiles to levels
                distributional_levels = (1 + percentiles) * np.array(
                    target_prediction_df["actual_closing_level"][idx]
                )
                distributional_levels = np.insert(
                    distributional_levels, 0, prob_below_zero
                )
                distributional_levels_list.append(
                    {f"{model_name}": distributional_levels}
                )

            # from list to dataframe
            distribution_df = pd.concat(
                [pd.DataFrame(x) for x in distributional_levels_list], axis=1
            ).T

            distribution_df = pd.DataFrame(
                distribution_df.values,
                columns=[
                    "prob_below_zero_ret",
                    "-2 sigma",
                    "-1 sigma",
                    "25%",
                    "50%",
                    "75%",
                    "+1 sigma",
                    "+2 sigma",
                ],
                index=distribution_df.index,
            )

            target_prediction_df = pd.concat(
                [target_prediction_df.iloc[seed_error:], distribution_df], axis=1
            )

            # convert from returns space to -1 [neg. return] and +1 [pos return]
            target_prediction_df["pred_binary_ret"] = np.where(
                target_prediction_df[f"pred_ret_next_fin rprt date"] < 0, -1, 1
            )
            target_prediction_df["actual_binary_ret"] = np.where(
                target_prediction_df[f"actual_ret_next_fin rprt date"] < 0, -1, 1
            )

            # load best features to S3 folder with model_name in filename
            aws_client.upload_object(
                bucket=args.bucket,
                filename=f"{filepath_out_aws}/results/{model_name}_tgt_pred_"
                f"num_best_feat={args.num_best_features}_"
                f"fit_frac={args.fit_fraction}_"
                f"seed_frac={args.seed_fraction}_"
                f"btest_hzon={args.backtest_horizon}_prds_"
                f"num_btest_periods={args.num_backtest_periods}_prds_"
                f"fcast_hzon=next_fin rprt date_"
                f"fcast_adj={args.forecast_adj[0]}_"
                f"tkr={name}_"
                f"{args.timestamp}.csv",
                fileobj=target_prediction_df.to_csv(index=True),
            )
            target_prediction_df.to_csv(
                f"{filepath_out_local}/results/{model_name}_tgt_pred_"
                f"num_best_feat={args.num_best_features}_"
                f"fit_frac={args.fit_fraction}_"
                f"seed_frac={args.seed_fraction}_"
                f"btest_hzon={args.backtest_horizon}_prds_"
                f"num_btest_periods={args.num_backtest_periods}_prds_"
                f"fcast_hzon=next_fin rprt date_"
                f"fcast_adj={args.forecast_adj[0]}_"
                f"tkr={name}_"
                f"{args.timestamp}.csv"
            )

            # append model backtest results to list
            multimodel_prediction_list.append({f"{model_name}": target_prediction_df})

            # convert list of temporal best features into dataframe of best features
            selector_best_features_df = pd.DataFrame(
                np.array(selector_best_features)[:, 0, :],
                index=y_df.index[-len(selector_best_features) :],
            )
            selector_best_features_df = selector_best_features_df.add_prefix(
                "best_feature_"
            )
            selector_best_features_df.reset_index(inplace=True)
            selector_best_features_df.rename(
                columns={"date": "forecast_date"}, inplace=True
            )
            selector_best_features_df.set_index(["forecast_date"], inplace=True)

            # load best features to S3 folder with model_name in filename
            aws_client.upload_object(
                bucket=args.bucket,
                filename=f"{filepath_out_aws}/best_features/{model_name}_"
                f"num_best_feat={args.num_best_features}_"
                f"fit_frac={args.fit_fraction}_"
                f"seed_frac={args.seed_fraction}_"
                f"btest_hzon={args.backtest_horizon}_prds_"
                f"fcast_hzon=next_fin rprt date_"
                f"fcast_adj={args.forecast_adj[0]}_"
                f"btest_dt={args.num_backtest_periods}_prds_"
                f"tkr={name}_"
                f"{args.timestamp}.csv",
                fileobj=selector_best_features_df.to_csv(index=True),
            )
            # save as csv on local
            selector_best_features_df.to_csv(
                f"{filepath_out_local}/best_features/{model_name}_"
                f"num_best_feat={args.num_best_features}_"
                f"fit_frac={args.fit_fraction}_"
                f"seed_frac={args.seed_fraction}_"
                f"btest_hzon={args.backtest_horizon}_prds_"
                f"fcast_hzon=next_fin rprt date_"
                f"fcast_adj={args.forecast_adj[0]}_"
                f"btest_dt={args.num_backtest_periods}_prds_"
                f"tkr={name}_"
                f"{args.timestamp}.csv"
            )

        # 5.a Score the models
        classification_diagnostics_list = []
        for model in range(0, len(multimodel_prediction_list)):
            for model_name, df in multimodel_prediction_list[model].items():
                # convert from returns space to -1 [neg. return] and +1 [pos return]
                df["pred_binary_ret"] = np.where(
                    df[f"pred_ret_next_fin rprt date"] < 0, -1, 1
                )
                df["actual_binary_ret"] = np.where(
                    df[f"actual_ret_next_fin rprt date"] < 0, -1, 1
                )
                # returns precision, recall scores for pos and neg ret predictions
                tn, fp, fn, tp = validation_metrics.confusion_matrix(
                    df["actual_binary_ret"], df["pred_binary_ret"]
                ).ravel()
                # fraction of predictions with correct directional accuracy
                accuracy_score = (tp + tn) / (tp + tn + fp + fn)
                # fraction of positive prediction with correct directional accuracy
                upside_precision_score = tp / (tp + fp)
                # fraction of positive return periods correctly predicted
                upside_recall_score = tp / (tp + fn)
                # fraction of negative prediction with correct directional accuracy
                downside_precision_score = tn / (tn + fn)
                # fraction of negative return periods correctly predicted
                downside_recall_score = tn / (tn + fp)
                frac_obs_zero = (
                    df[f"pred_ret_next_fin rprt date"].diff() == 0
                ).sum() / len(df)

                diagnostics_dict = {
                    model_name: {
                        "accuracy_score": accuracy_score,
                        "upside_precision_score": upside_precision_score,
                        "upside_recall_score": upside_recall_score,
                        "downside_precision_score": downside_precision_score,
                        "downside_recall_score": downside_recall_score,
                        "frac_model_pred_ret=0": frac_obs_zero,
                    }
                }
                classification_diagnostics_list.append(diagnostics_dict)

        # 6a. load classification_diagnostics across all models
        df = pd.concat(
            [pd.DataFrame(x) for x in classification_diagnostics_list], axis=1
        ).T
        LOGGER.info(
            "loading to S3 classification backtester diagnostics across all models"
        )
        # load val to S3 folder with model_name in filename
        aws_client.upload_object(
            bucket=args.bucket,
            filename=f"{filepath_out_aws}/classification_diagnostics/"
            f"classification_diagnostics_"
            f"best_feat={args.num_best_features}_"
            f"btest_hzon={args.backtest_horizon}_prds_"
            f"fcast_hzon=next_fin rprt date_"
            f"btest_dt={args.num_backtest_periods}_prds_"
            f"tkr={name}_"
            f"{args.timestamp}.csv",
            fileobj=df.to_csv(index=True),
        )
        # save as csv on local
        df.to_csv(
            f"{filepath_out_local}/classification_diagnostics/"
            f"classification_diagnostics_"
            f"best_feat={args.num_best_features}_"
            f"btest_hzon={args.backtest_horizon}_prds_"
            f"fcast_hzon=next_fin rprt date_"
            f"btest_dt={args.num_backtest_periods}_prds_"
            f"tkr={name}_"
            f"{args.timestamp}.csv"
        )
