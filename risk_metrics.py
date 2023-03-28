from typing import Dict, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel
from riskfolio import RiskFunctions


class ModelConfig(BaseModel):
    class Config:
        arbitrary_types_allowed = True


class risk_metrics_selector_config(ModelConfig):
    backtest_binary_pred: Union[pd.DataFrame, np.ndarray]
    actual_ret: Union[pd.DataFrame, np.ndarray]


class risk_metrics:
    r"""
    Class that creates object with all properties needed to calculate risk metrics &
    risk contributions across risk/return optimized portfolios

    Parameters
    ----------
    """

    def __init__(
        self,
        config: risk_metrics_selector_config,
    ):
        self.backtest_binary_pred = config.backtest_binary_pred
        self.actual_ret = config.actual_ret

    def backtester_returns(self, benchmark: bool = None) -> pd.DataFrame:
        if benchmark:
            rets = np.array(self.actual_ret)
        else:
            rets = np.where(
                self.backtest_binary_pred == 1, self.actual_ret, -self.actual_ret
            )

        return rets

    def port_opt_efficient_frontier_returns(
        self, efficient_frontier_wgts: pd.DataFrame, asset_level_ret: pd.DataFrame
    ) -> pd.DataFrame:
        if isinstance(efficient_frontier_wgts, pd.DataFrame):
            efficient_frontier_wgts = efficient_frontier_wgts.loc[
                :, ~efficient_frontier_wgts.columns.str.contains("^Unnamed")
            ]
            efficient_frontier_wgts = efficient_frontier_wgts.values

        if isinstance(asset_level_ret, pd.DataFrame):
            asset_level_ret = asset_level_ret.loc[
                :, ~asset_level_ret.columns.str.contains("^Unnamed")
            ]
            asset_level_ret = asset_level_ret.values

        if efficient_frontier_wgts.shape[0] < efficient_frontier_wgts.shape[1]:
            efficient_frontier_wgts = efficient_frontier_wgts.T

        if efficient_frontier_wgts.shape[1] != asset_level_ret.shape[0]:
            asset_level_ret = asset_level_ret.T

        efficient_frontier_rets = efficient_frontier_wgts @ asset_level_ret

        return efficient_frontier_rets

    def risk_measurement(
        self,
        returns,
        alpha: float,
    ) -> Dict:
        # initialize the risk metrics class object
        # estimate historical sim returns of port assuming opt weights through time

        VaR_Hist = RiskFunctions.VaR_Hist(returns, alpha=alpha)
        CVaR_Hist = RiskFunctions.CVaR_Hist(returns, alpha=alpha)
        wr = RiskFunctions.WR(returns)
        mdd = RiskFunctions.MDD_Abs(returns)
        add = RiskFunctions.ADD_Abs(returns)

        return {
            "VaR": VaR_Hist,
            "CVaR": CVaR_Hist,
            "wr": wr,
            "mdd": mdd,
            "add": add,
        }
