import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from matplotlib.lines import Line2D
import time
import ast

import warnings
warnings.filterwarnings('ignore')

start = time.perf_counter()

#-----------------------------------------------------------------------------------------------------------
# FUNCTION : PARSE THE DICT
def parse_dict_cell(x):
    if isinstance(x, dict):
        return x

    if pd.isna(x):
        return {}

    if not isinstance(x, str):
        raise TypeError(f"Unexpected type: {type(x)} | value={x}")

    s = x.strip()

    if s == "":
        return {}

    try:
        parsed = eval(
            s,
            {
                "__builtins__": {},
                "np": np,
                "numpy": np,
                "array": np.array,
            },
        )
    except Exception as e:
        raise ValueError(f"Cannot parse OLS_hedge cell:\n{x}") from e

    if not isinstance(parsed, dict):
        raise TypeError(f"Parsed value is not a dict. Got {type(parsed)} | value={parsed}")

    return parsed


# FUNCTION : SPECIFIC PAIR TRADING VISUALIZATION
def pair_trading_visualize(df,
                           z_window,
                           entry_z: float = 2.0,
                           exit_z: float = 0):
    df = df.copy()
    # create canvas for 2 charts
    S1 = df.columns[0]
    S2 = df.columns[1]
    fig, facet = plt.subplots(3, 1,
                              figsize=[16, 8],
                              sharex=True,
                              gridspec_kw={'height_ratios': [2,2,1],
                                           'hspace': 0.05
                                           }
                              )
    ax1 = facet[0]
    ax2 = facet[1]
    ax3 = facet[2]
    # price line
    ax1.plot(df.index,
             df[S1],
             '-',
             color='grey',
             lw=0.8,
             label=S1
             )

    ax1.plot(df.index,
             df[S2],
             '-',
             color='blue',
             lw=0.8,
             label=S2
             )

    # long opportunities
    ax1.plot(df.query('long_signal == 1').index,
             df.query('long_signal == 1')[S1],
             '^',
             color='green',
             markersize=7
             )

    ax1.plot(df.query('long_signal == 1').index,
             df.query('long_signal == 1')[S2],
             'v',
             color='red',
             markersize=7
             )

    # short opportunities
    ax1.plot(df.query('short_signal == 1').index,
             df.query('short_signal == 1')[S1],
             'v',
             color='red',
             markersize=7
             )

    ax1.plot(df.query('short_signal == 1').index,
             df.query('short_signal == 1')[S2],
             '^',
             color='green',
             markersize=7
             )

    ax2.plot(df.index,
             df['cum_strategy_return'],
             '-',
             color='green',
             lw=1.2,
             label='Cumulative return'
             )

    ax2.plot(df.index,
             df['roll_max'],
             '-',
             color='red',
             lw=1,
             label='Roll max return'
             )

    # z-score
    ax3.plot(df.index,
             df['zscore'],
             '-',
             color='green',
             lw=1,
             label='Z-score'
             )

    # lower bound
    ax3.axhline(y=-entry_z,
                color='black',
                lw=0.5
                )

    # middle line
    ax3.axhline(y=exit_z,
                color='black',
                lw=0.5
                )

    # upper bound
    ax3.axhline(y=entry_z,
                color='black',
                lw=0.5
                )

    # long opportunities
    ax3.plot(df.query('long_signal == 1').index,
             df.query('long_signal == 1')['zscore'],
             '^',
             color='green',
             markersize=7
             )

    # short opportunities
    ax3.plot(df.query('short_signal == 1').index,
             df.query('short_signal == 1')['zscore'],
             'v',
             color='red',
             markersize=7
             )

    ax1.legend(bbox_to_anchor=(0.01, 0.95),
               loc='upper left',
               fontsize=10,
               frameon=False)

    ax2.legend(bbox_to_anchor=(0.01, 0.95),
               loc='upper left',
               fontsize=10,
               frameon=False)

    ax3.legend(bbox_to_anchor=(0.001, 1.1),
               loc='upper left',
               fontsize=10,
               frameon=False)

    # ax1.set_yticks(np.arange(20, 60, 5))
    # ax2.set_yticks(np.arange(0.5, 1.3, 0.1))
    # ax3.set_yticks(np.arange(0, 110, 20))

    ax1.grid(None)
    ax2.grid(None)
    ax3.set_title(f'Window: {z_window}',fontsize=8, x=0.01, y=0.8, ha='left', va='top')

    plt.show()

# -----------------------------------------------------------------------------------------------------------
# FUNCTION : SUMMARY VISUALIZATION
def plot_cagr(
    x,
    title: str = "CAGR Histogram",
    x_label: str = "CAGR",
    figsize: tuple = (9, 5),
    density: bool = False,
):
    """
    Plot histogram with custom CAGR bins:

    <0
    0% - 10%
    10% - 20%
    20% - 30%
    30% - 40%
    40% - 50%
    >50%

    Parameters
    ----------
    x : array-like
        CAGR data in decimal format.
        Example: 0.15 means 15%.
    density : bool
        If False, y-axis is frequency/count.
        If True, y-axis is density.
    """

    x = pd.Series(x, dtype="float64")
    x = x.replace([np.inf, -np.inf], np.nan).dropna()

    if x.empty:
        raise ValueError("Input x has no valid numeric values.")

    bins = [-np.inf, 0, 0.10, 0.20, 0.30, 0.40, 0.50, np.inf]

    labels = [
        "<0%",
        "0%–10%",
        "10%–20%",
        "20%–30%",
        "30%–40%",
        "40%–50%",
        ">50%",
    ]

    binned = pd.cut(
        x,
        bins=bins,
        labels=labels,
        right=False
    )

    counts = binned.value_counts(sort=False)

    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(
        counts.index.astype(str),
        counts.values
    )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Frequency")

    ax.grid(axis="y", alpha=0.3)

    for i, value in enumerate(counts.values):
        ax.text(
            i,
            value,
            str(value),
            ha="center",
            va="bottom"
        )

    plt.tight_layout()
    plt.show()

    return counts

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_sharpe(
    x,
    title: str = "Sharpe Ratio Histogram",
    x_label: str = "Sharpe Ratio",
    figsize: tuple = (9, 5),
):
    """
    Plot histogram with custom Sharpe ratio bins:

    < 0
    0 - 1
    1 - 2
    2 - 3
    > 3

    Parameters
    ----------
    x : array-like
        Sharpe ratio data.
    """

    x = pd.Series(x, dtype="float64")
    x = x.replace([np.inf, -np.inf], np.nan).dropna()

    if x.empty:
        raise ValueError("Input x has no valid numeric values.")

    bins = [-np.inf, 0, 1, 2, 3, np.inf]

    labels = [
        "<0",
        "0–1",
        "1–2",
        "2–3",
        ">3",
    ]

    binned = pd.cut(
        x,
        bins=bins,
        labels=labels,
        right=False
    )

    counts = binned.value_counts(sort=False)

    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(
        counts.index.astype(str),
        counts.values
    )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("No. of pairs")

    ax.grid(axis="y", alpha=0.3)

    for i, value in enumerate(counts.values):
        ax.text(
            i,
            value,
            str(value),
            ha="center",
            va="bottom"
        )

    plt.tight_layout()
    plt.show()

    return counts
#-----------------------------------------------------------------------------------------------------------
# FUNCTION : COMPUTE HEDGING PARAMETERS
def OLS_regression(X,Y):
    # Ensure X has a name for indexing later
    if not hasattr(X, 'name') or X.name is None:
        X.name = 'x_variable'

    y = Y.to_numpy(dtype=float)
    x = X.to_numpy(dtype=float)

    X_with_const = sm.add_constant(X)
    model = sm.OLS(Y, X_with_const)
    fitted_model = model.fit()

    # compute hedge parameters
    alpha = fitted_model.params['const']
    beta = fitted_model.params[X.name]
    state_cov = fitted_model.cov_params().to_numpy()

    # compute z_window
    residuals = fitted_model.resid
    model_AR1 = AutoReg(residuals, lags=1).fit()
    phi = model_AR1.params.iloc[1]
    z_window = 15 if phi<0 or phi>1 else round(np.log(0.5)/np.log(abs(phi)),0).astype(int)

    output = {'alpha':alpha,
              'beta':beta,
              'phi':phi,
              'z_window':z_window,
              'state_cov':state_cov}

    return output

# -----------------------------------------------------------------------------------------------------------
# FUNCTION : KALMAN FILTER
def compute_Kalman_filter(
        x,
        y,
        alpha,
        beta,
        state_cov,
        alpha_noise=1e-8,
        beta_noise=1e-6,
        obs_noise=1e-4
):
    x = np.asarray(x)
    y = np.asarray(y)

    # Initializing
    n = len(y)
    alphas = np.empty(n)
    betas = np.empty(n)
    I = np.eye(2)

    # starting state (from OLS regression)
    theta = np.array([alpha, beta], dtype=float)
    P = np.diag(np.diag(np.asarray(state_cov, dtype=float)))

    # state noise: controls alpha/beta drift
    Q = np.diag([alpha_noise, beta_noise])

    # Observation noise
    R = float(obs_noise)

    for t in range(n):
        #observation
        F = np.array([1.0, x[t]])

        # Prediction step
        theta_pre = theta
        P_pre = P + Q

        # Observation prediction
        y_pred = F @ theta_pre
        innovation = y[t] - y_pred

        # innovation var
        S = F @ P_pre @ F + R
        S = max(S, 1e-12)

        # Kalman gain
        K = P_pre @ F / S

        # Update step
        theta = theta_pre + K * innovation
        theta = np.asarray(theta).flatten()
        KF = np.outer(K, F)
        P =  (
                (I - KF) @ P_pre @ (I - KF).T
                + R * np.outer(K, K)
        )
        P = np.diag(np.diag(P))

        # pred_state_cov
        alphas[t] = theta[0]
        betas[t] = theta[1]

    return alphas, betas

# -----------------------------------------------------------------------------------------------------------
# FUNCTION : TRADING STRATEGY
def pair_trading_execution(
    df:pd.DataFrame,
    alpha,
    beta,
    z_window,
    state_cov,
    Kalman_filter:bool,
    entry_z=2.0,
    exit_z=0.0,
    stop_z=4.0,
) -> dict:

    df = df.copy()
    df['daily_return_Y'] = df.iloc[:,0].pct_change()
    df['daily_return_X'] = df.iloc[:,1].pct_change()
    df.dropna(inplace=True)

    y = df.iloc[:,0].to_numpy(dtype=float)
    x = df.iloc[:,1].to_numpy(dtype=float)
    n = len(y)

    alphas = np.zeros(n)
    betas = np.zeros(n)

    if Kalman_filter == True:
        alphas, betas = compute_Kalman_filter(x=x,y=y,alpha=alpha,beta=beta,state_cov=state_cov)
    else:
        alphas = np.zeros(n) + alpha
        betas = np.zeros(n) + beta

    df['alpha'] = alphas
    df['beta'] = betas

    # compute z-score
    spreads = y - alphas - x * betas
    df['spread'] = spreads
    df['spread_mean'] = df['spread'].rolling(z_window).mean().shift(1)
    df['spread_std'] = df['spread'].rolling(z_window).std(ddof=1).shift(1)
    df['zscore'] = (df['spread'] - df['spread_mean']) / df['spread_std']

    # detect signal
    df['signal'] = \
        (
            np
            .where(
                df['zscore'] <= -entry_z,
                1, np.nan
            )
        )

    df['signal'] = \
        (
            np
            .where(
                df['zscore'] >= entry_z,
                -1, df['signal']
            )
        )

    df['distance'] = \
        (
            np.where(
                df['zscore'] > 0,
                df['zscore'] - exit_z,
                df['zscore'] + exit_z
            )
        )

    df['signal'] = \
        (
            np
            .where(
                (df['zscore'] > -entry_z)
                &
                (df['zscore'] < entry_z)
                &
                (df['distance'] * df['distance'].shift(1) < 0),
                0, df['signal'],
            )
        )

    df['signal'] = \
        (
            np
            .where(
                (df['zscore'].shift(1).abs() > entry_z)
                &
                (df['zscore'].abs() > stop_z),
                0, df['signal'],
            )
        )

    df['signal'] = \
        (
            df['signal']
            .ffill()
            .fillna(0)
        )

    # Execute next bar: position at t is based on signal from t-1.
    df["position"] = df["signal"].shift(1).fillna(0.0)

    # identify trading opportunities
    df['long'] = np.where(df['position'] == 1, 1, 0)
    df['long_signal'] = df['long'].diff().shift(-1)

    df['short'] = np.where(df['position'] == -1, 1, 0)
    df['short_signal'] = df['short'].diff().shift(-1)

    df['beta_pnl'] = df['beta'].shift(1)
    df['spread_daily_return'] = df['daily_return_Y'] - df['beta_pnl'] * df['daily_return_X']
    df['daily_strategy_return'] = df['spread_daily_return']*df['position']
    df["daily_strategy_return"] = df["daily_strategy_return"].fillna(0.0)

    df['cum_strategy_return'] = (1 + df['daily_strategy_return']).cumprod()
    df['roll_max'] = df['cum_strategy_return'].cummax()
    df['dd'] = (df['roll_max'] - df['cum_strategy_return']) / df['roll_max']
    dd_periods = \
        (
            np
            .diff(
                np.append(df.query('dd == 0').index,
                          df.index[-1:]
                          )
            )
        ) / np.timedelta64(1, 'D')

    horizon = (df.index[-1] - df.index[0]).days

    # metrics calculation
    cagr = df['cum_strategy_return'].iloc[-1] ** (365/horizon) - 1
    cum_return = df['cum_strategy_return'].iloc[-1] - 1
    mean = df['daily_strategy_return'].mean()
    std = df['daily_strategy_return'].std()
    sharpe = np.nan if std == 0 else np.sqrt(252) * mean / std
    max_dd = df['dd'].max()
    longest_dd = dd_periods.max()
    no_trade = (df['long_signal']==1).sum() + (df['short_signal']==1).sum()

    metrics = \
        {
            'sharpe': sharpe,
            'cagr': cagr,
            'cum_return': cum_return,
            'max_dd': max_dd,
            'longest_dd': longest_dd,
            'no_trade' : no_trade
        }

    return metrics,df

# -----------------------------------------------------------------------------------------------------------
# LOADING DATA
dataset = pd.read_csv('SP500.csv')
tickers_info = pd.read_csv('tickers_info.csv')
clustered_pairs = pd.read_csv('clustered_pairs.csv')

dataset.dropna(axis=1,inplace=True)
dataset["Date"] = pd.to_datetime(dataset["Date"])
dataset = dataset.set_index("Date").sort_index()
clustered_pairs.set_index('pair',inplace=True)

dataset_phase2 = dataset.loc[dataset.index.year == 2023]
dataset_phase3 = dataset.loc[dataset.index.year == 2024]
dataset_phase4 = dataset.loc[dataset.index.year == 2025]

# -----------------------------------------------------------------------------------------------------------
# HEDGING PARAMETERS FROM CLUSTERED PAIRS

params_phase2 = []
for pair in clustered_pairs.index:

    pair_idx = list(ast.literal_eval(pair))

    S1 = pair_idx[0]
    S2 = pair_idx[1]

    X = dataset_phase2[S2]
    Y = dataset_phase2[S1]

    OLS_output = OLS_regression(X, Y)
    params_phase2.append(OLS_output)


pairs_phase2 = pd.DataFrame({'pair':clustered_pairs.index,
                             'params': params_phase2
                               }
                              )
pairs_phase2.set_index('pair',inplace=True)
pairs_phase2['params'] = pairs_phase2['params'].apply(parse_dict_cell)

# -----------------------------------------------------------------------------------------------------------
# RUNNING FOR ALL CLUSTERED PAIRS
perf_phase3 = {}
for i, pair in enumerate(pairs_phase2.index):
    pair_idx = list(ast.literal_eval(pair))
    data = dataset_phase3[[pair_idx[0], pair_idx[1]]]
    params = pairs_phase2.iloc[i]['params']
    alpha = params['alpha']
    beta = params['beta']
    phi = params['phi']
    z_window = params['z_window']
    state_cov = params['state_cov']
    metrics, df = pair_trading_execution(df=data,
                                         alpha=alpha, beta=beta, z_window=z_window,state_cov=state_cov,
                                         Kalman_filter=False
                                         )
    perf_phase3[pair] = metrics

perf_phase3K = {}
for i, pair in enumerate(pairs_phase2.index):
    pair_idx = list(ast.literal_eval(pair))
    data = dataset_phase3[[pair_idx[0], pair_idx[1]]]
    params = pairs_phase2.iloc[i]['params']
    alpha = params['alpha']
    beta = params['beta']
    phi = params['phi']
    z_window = params['z_window']
    state_cov = params['state_cov']
    metrics, df = pair_trading_execution(df=data,
                                         alpha=alpha, beta=beta, z_window=z_window,state_cov=state_cov,
                                         Kalman_filter=True
                                         )
    perf_phase3K[pair] = metrics

# -----------------------------------------------------------------------------------------------------------
# BACKTESTING
params_phase3 = []
for pair in clustered_pairs.index:

    pair_idx = list(ast.literal_eval(pair))

    S1 = pair_idx[0]
    S2 = pair_idx[1]

    X = dataset_phase3[S2]
    Y = dataset_phase3[S1]

    OLS_output = OLS_regression(X, Y)
    params_phase3.append(OLS_output)

pairs_phase3 = pd.DataFrame({'pair':clustered_pairs.index,
                             'params': params_phase3
                               }
                              )
pairs_phase3.set_index('pair',inplace=True)
pairs_phase3['params'] = pairs_phase3['params'].apply(parse_dict_cell)

perf_phase4 = {}
for i, pair in enumerate(pairs_phase3.index):
    pair_idx = list(ast.literal_eval(pair))
    data = dataset_phase4[[pair_idx[0], pair_idx[1]]]
    params = pairs_phase3.iloc[i]['params']
    alpha = params['alpha']
    beta = params['beta']
    phi = params['phi']
    z_window = params['z_window']
    state_cov = params['state_cov']
    metrics, df = pair_trading_execution(df=data,
                                         alpha=alpha, beta=beta, z_window=z_window,state_cov=state_cov,
                                         Kalman_filter=False
                                         )
    perf_phase4[pair] = metrics

perf_phase4K = {}
for i, pair in enumerate(pairs_phase3.index):
    pair_idx = list(ast.literal_eval(pair))
    data = dataset_phase4[[pair_idx[0], pair_idx[1]]]
    params = pairs_phase3.iloc[i]['params']
    alpha = params['alpha']
    beta = params['beta']
    phi = params['phi']
    z_window = params['z_window']
    state_cov = params['state_cov']
    metrics, df = pair_trading_execution(df=data,
                                         alpha=alpha, beta=beta, z_window=z_window,state_cov=state_cov,
                                         Kalman_filter=True
                                         )
    perf_phase4K[pair] = metrics

# -----------------------------------------------------------------------------------------------------------
# TESTING FOR A SPECIFIC PAIR
# pair = "('MRSH', 'WMT')"
# pair_idx = pairs_phase2.index.get_loc(pair)
# pair_list = list(ast.literal_eval(pair))
# data = dataset_phase3[[pair_list[0], pair_list[1]]]
# params = pairs_phase2.iloc[pair_idx]['params']
# alpha = params['alpha']
# beta = params['beta']
# z_window = params['z_window']
# state_cov = params['state_cov']
#
# metrics, df = pair_trading_execution(df=data,
#                                      alpha=alpha, beta=beta, z_window=z_window,state_cov=state_cov,
#                                      Kalman_filter=False
#                                      )
#
#
# # print(df[df.index.month ==1][['spread','zscore','position','daily_strategy_return']])
# # df.to_csv('test.csv')
# pair_trading_visualize(df,z_window)

# -----------------------------------------------------------------------------------------------------------
# ANALYTICS
summary = pd.DataFrame([perf_phase3,
                         perf_phase4,
                         perf_phase3K,
                         perf_phase4K],
                       index = ['phase3','phase4','phase3K','phase4K'])

summary = summary.T
summary['phase3'] = summary['phase3'].apply(parse_dict_cell)
summary['phase4'] = summary['phase4'].apply(parse_dict_cell)
summary['phase3K'] = summary['phase3K'].apply(parse_dict_cell)
summary['phase4K'] = summary['phase4K'].apply(parse_dict_cell)

summary['sharpe_3'] = summary['phase3'].str['sharpe']
summary['sharpe_4'] = summary['phase4'].str['sharpe']
summary['sharpe_3K'] = summary['phase3K'].str['sharpe']
summary['sharpe_4K'] = summary['phase4K'].str['sharpe']

summary['cagr_3'] = summary['phase3'].str['cagr']
summary['cagr_4'] = summary['phase4'].str['cagr']
summary['cagr_3K'] = summary['phase3K'].str['cagr']
summary['cagr_4K'] = summary['phase4K'].str['cagr']

top20 = summary.nlargest(20, "sharpe_3")
top20K = summary.nlargest(20, "sharpe_3K")

# print(top20[['cagr_3','cagr_3K','sharpe_3','sharpe_3K']])
# print(top20K[['cagr_3','cagr_3K','sharpe_3','sharpe_3K']])

backtest_pairs = summary.query('sharpe_3 > 0 & sharpe_3K > 0')
plot_data = backtest_pairs[['sharpe_3K','sharpe_4K']]
a = (plot_data['sharpe_3K'] >= 1).sum()
b = (
    (plot_data["sharpe_3K"] >= 1) &
    (plot_data["sharpe_4K"] > 0)
).sum()
print(b/a)
# plot_sharpe(summary['sharpe_3K'])
# plot_sharpe(backtest_pairs['sharpe_4K'])

# plot_cagr(summary_phase3['cagr'])
# plot_sharpe(summary_phase3['sharpe'])

# -----------------------------------------------------------------------------------------------------------

def plot_backtest(
    df: pd.DataFrame,
    point_size: float = 25,
    figsize: tuple = (10, 6),
    title: str = "Backtest Scatter Plot",
    x_label: str | None = None,
    y_label: str | None = None,
):
    """
    Plot first column of df as x and second column as y.

    Point color:
        green if y >= 0
        red if y < 0

    Background:
        green area if y >= 0
        red area if y < 0
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if df.shape[1] < 2:
        raise ValueError("df must contain at least two columns.")

    data = df.iloc[:, :2].copy()
    data.columns = ["x", "y"]

    data["x"] = pd.to_numeric(data["x"], errors="coerce")
    data["y"] = pd.to_numeric(data["y"], errors="coerce")

    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    if data.empty:
        raise ValueError("No valid numeric x/y data to plot.")

    x = data["x"].to_numpy(dtype=float)
    y = data["y"].to_numpy(dtype=float)

    colors = np.where(y >= 0, "green", "red")

    fig, ax = plt.subplots(figsize=figsize)

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    x_padding = 0.05 * (x_max - x_min) if x_max > x_min else 1.0
    y_padding = 0.10 * (y_max - y_min) if y_max > y_min else 1.0

    x_lower = x_min - x_padding
    x_upper = x_max + x_padding

    y_lower = min(y_min - y_padding, -0.1)
    y_upper = max(y_max + y_padding, 0.1)

    ax.axhspan(y_lower, 0, color="red", alpha=0.08)
    ax.axhspan(0, y_upper, color="green", alpha=0.06)

    ax.scatter(
        x,
        y,
        c=colors,
        s=point_size,
        alpha=0.8,
        edgecolors="black",
        linewidths=0.3
    )

    ax.axhline(
        0,
        color="black",
        linestyle="--",
        linewidth=1.2
    )

    legend_elements = [
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            label="y >= 0",
            markerfacecolor="green",
            markeredgecolor="black",
            markersize=7
        ),
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            label="y < 0",
            markerfacecolor="red",
            markeredgecolor="black",
            markersize=7
        ),
    ]

    ax.legend(handles=legend_elements)

    ax.set_xlim(x_lower, x_upper)
    ax.set_ylim(y_lower, y_upper)

    ax.set_title(title)
    ax.set_xlabel('sharpe | Testing')
    ax.set_ylabel('sharpe | Backtest')

    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return fig, ax

# plot_backtest(plot_data)

end = time.perf_counter()
elapsed_time = end - start
print(f"Total running time: {elapsed_time:.4f} seconds")

