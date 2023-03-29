# investment-research-lab

## process for predicting quarterly stock price returns using fundamental data of the given stock

## Step 1: api the stock price data  
python scripts/polygon_collection.py \`  
--days_lookback 9000 \`  


## Step 2: create the necessary data structure consisting of both fundamental data [predictors] and stock price data [target]
python scripts/get_data.py \`  
--indicator_model "fundamental" \`  
--universe_partition "[0, 1]" \`  
--bucket_out "investment-research-lab" \`   


## Step 3: run the backtest
python scripts/backtester.py \`  
--bucket "investment-research-lab" \`  
--indicator_model "fundamental" \`  
--estimator_model_class "ensemble" \`  
--num_best_features 3 \`  
--fit_fraction 1 \`  
--seed_fraction 0.4 \`  
--target "close" \`  
--backtest_horizon 20 \`  
--num_backtest_periods 40 \`  
--forecast_horizon 1 \`  
--universe_partition "[0, 1]" \`  
--forecast_adj "False" \`  
--timestamp "20230305145929" \`    


## Step 4: run the backtest pnl summary across stocks
python scripts/summary_pnl_stats.py \`  
--bucket "investment-research-lab" \`  
--indicator_model "fundamental" \`  
--summary_type "backtest" \`  
--num_best_features 3 \`  
--quantile_list "[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]" \`  
--fit_fraction 1 \`  
--seed_fraction 0.4 \`  
--target_definition "['close']" \`  
--backtest_horizon "['20']" \`  
--num_backtest_periods "['40']" \`  
--forecast_horizon "next_fin rprt date" \`  
--universe_partition 1 \`  
--forecast_adj "['False']" \`  
--alpha 0.05 \`   
--timestamp "20230305145929" \`    
