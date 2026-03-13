import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tushare as ts
import warnings
from datetime import datetime, timedelta
import base64
from io import BytesIO
from tabulate import tabulate
import mimetypes
from pathlib import Path
import os
import email_sender_v2
import a_passwards as pw

# 设置中文字体，确保中文正常显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
warnings.filterwarnings('ignore')

# 设置tushare token
ts.set_token(pw.TUSHARE_TOKEN)
pro = ts.pro_api()

# 定义资产代码
ASSET_CODES = {
    'Stock': '000300.SH',  # 沪深300
    'Bond': '511010.SH',  # 国债ETF
    'Gold': '518880.SH'  # 黄金ETF
}

ASSET_NAMES = {
    '000300.SH': '沪深300',
    '511010.SH': '国债ETF',
    '518880.SH': '黄金ETF'
}

# 邮件发送配置
SENDGRID_API_KEY = ""  # 请在这里填入您的SendGrid API Key
SENDER_EMAIL = ""  # 请在这里填入您的发件人邮箱
RECIPIENTS = [
    # "example1@email.com",
    # "example2@email.com",
]

# 确定初始资产权重
weights = {'Stock': 0.4, 'Bond': 0.3, 'Gold': 0.3}

# 定义阈值
max_threshold = 0.1
min_threshold = 0.1


# 从tushare获取数据
def fetch_data_from_tushare(start_date='20210101', end_date=datetime.today().strftime('%Y%m%d')):
    """
    从tushare获取数据
    """
    print("正在从tushare获取数据...")

    all_data = {}

    for asset_type, ts_code in ASSET_CODES.items():
        print(f"获取{ASSET_NAMES[ts_code]}数据...")

        if asset_type == 'Stock':
            # 指数数据
            data = pro.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        else:
            # ETF基金数据
            data = pro.fund_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)

        if len(data) == 0:
            print(f"警告: {ASSET_NAMES[ts_code]} 无数据")
            continue

        data['trade_date'] = pd.to_datetime(data['trade_date'])
        data = data.set_index('trade_date').sort_index()

        # 确保有收盘价数据
        if 'close' in data.columns:
            all_data[asset_type] = data['close']
        else:
            print(f"警告: {ASSET_NAMES[ts_code]} 没有close列")

    return all_data


# 方式1：从Excel文件读取数据（如果已有数据文件）
try:
    df = pd.read_excel('Final Data.xlsx')
    df = df.set_index('date')
    print("从Excel文件加载数据成功")
except:
    print("无法从Excel文件加载数据，尝试从tushare获取...")
    # 方式2：从tushare获取数据
    try:
        all_data = fetch_data_from_tushare()

        if len(all_data) == 0:
            raise ValueError("从tushare获取的数据为空")

        # 合并数据 - 使用内连接确保所有资产都有数据
        df = pd.DataFrame(all_data)
        df = df.dropna()

        print(f"从tushare获取数据成功，共 {len(df)} 行数据")
        print("数据时间范围:", df.index.min(), "至", df.index.max())
        print(df.head())

    except Exception as e:
        print(f"从tushare获取数据失败: {e}")
        print("创建示例数据用于演示...")
        # 创建示例数据
        dates = pd.date_range(start='2015-01-01', end='2025-11-01', freq='D')
        np.random.seed(42)

        stock_returns = np.random.randn(len(dates)) * 0.01
        bond_returns = np.random.randn(len(dates)) * 0.002
        gold_returns = np.random.randn(len(dates)) * 0.005

        stock_prices = 100 * (1 + np.cumsum(stock_returns))
        bond_prices = 100 * (1 + np.cumsum(bond_returns))
        gold_prices = 100 * (1 + np.cumsum(gold_returns))

        df = pd.DataFrame({
            'Stock': stock_prices,
            'Bond': bond_prices,
            'Gold': gold_prices
        }, index=dates)

        # 只保留工作日（模拟交易日）
        df = df[df.index.dayofweek < 5]
        print("使用示例数据继续运行...")

# 计算每列的日收益率
returns = df.pct_change().dropna()

print(f"收益率数据形状: {returns.shape}")
print("数据时间范围:", returns.index.min(), "至", returns.index.max())

# 检查是否有足够的数据
if len(returns) == 0:
    print("错误：没有足够的收益率数据")
    exit()


# 定义回测结果类
class BacktestResult:
    def __init__(self, returns, weights, max_threshold=0.1, min_threshold=0.1):
        self.returns = returns
        self.weights = weights
        self.max_threshold = max_threshold
        self.min_threshold = min_threshold
        self.dates = returns.index
        self.portfolio_values = pd.Series(index=self.dates)
        self.daily_returns = pd.Series(index=self.dates[1:])
        self.daily_weights = pd.DataFrame(index=self.dates, columns=returns.columns)
        self.asset_returns = pd.DataFrame(index=self.dates, columns=returns.columns)
        self.rebalance_dates = []  # 记录调仓日期
        self.rebalance_flags = pd.Series(False, index=self.dates)  # 记录每天是否调仓

    def calculate_weights(self, portfolio):
        total_value = portfolio.sum()
        return {asset: portfolio[asset] / total_value for asset in portfolio.index}

    def rebalance_portfolio(self, portfolio, current_weights, date):
        """重新平衡投资组合，返回调整后的组合和是否调仓的标志"""
        needs_rebalance = False
        rebalance_reason = ""

        # 检查每个资产是否超出阈值
        for asset in portfolio.index:
            if current_weights[asset] >= self.weights[asset] + self.max_threshold:
                needs_rebalance = True
                rebalance_reason = f"{asset}权重超过上限阈值"
                break
            elif current_weights[asset] <= self.weights[asset] - self.min_threshold:
                needs_rebalance = True
                rebalance_reason = f"{asset}权重低于下限阈值"
                break

        # 如果需要调仓，调整到目标权重
        if needs_rebalance:
            new_weights = pd.Series(self.weights, index=portfolio.index)
            portfolio = new_weights * portfolio.sum()
            self.rebalance_dates.append(date)
            self.rebalance_flags[date] = True
            return portfolio, True, rebalance_reason

        return portfolio, False, ""

    def get_next_day_weights(self, current_weights):
        """根据当前权重和目标权重，判断下一日是否需要调仓，并返回下一日建议权重"""
        # 检查是否需要调仓
        needs_rebalance = False
        rebalance_reason = ""

        for asset in current_weights.index:
            if current_weights[asset] >= self.weights[asset] + self.max_threshold:
                needs_rebalance = True
                rebalance_reason = f"{asset}权重({current_weights[asset]:.1%})超过目标权重({self.weights[asset]:.0%}+{self.max_threshold:.0%}={self.weights[asset] + self.max_threshold:.1%})"
                break
            elif current_weights[asset] <= self.weights[asset] - self.min_threshold:
                needs_rebalance = True
                rebalance_reason = f"{asset}权重({current_weights[asset]:.1%})低于目标权重({self.weights[asset]:.0%}-{self.min_threshold:.0%}={self.weights[asset] - self.min_threshold:.1%})"
                break

        if needs_rebalance:
            # 如果需要调仓，返回目标权重
            next_weights = pd.Series(self.weights, index=current_weights.index)
            return next_weights, True, rebalance_reason
        else:
            # 如果不需要调仓，保持当前权重
            return current_weights, False, "各资产权重均在目标范围内"

    def run_backtest(self):
        """运行回测并记录每日持仓和表现"""
        # 初始化
        self.portfolio_values.iloc[0] = 1
        portfolio = self.portfolio_values.iloc[0] * pd.Series(self.weights)
        self.daily_weights.iloc[0] = pd.Series(self.weights)
        self.rebalance_flags.iloc[0] = True  # 初始建仓视为调仓
        self.rebalance_dates.append(self.dates[0])

        # 记录初始日的资产收益（为0）
        self.asset_returns.iloc[0] = 0

        # 逐日回测
        for i in range(1, len(self.dates)):
            date = self.dates[i]
            prev_date = self.dates[i - 1]

            # 计算当日各资产收益
            asset_ret = self.returns.loc[date]
            self.asset_returns.loc[date] = asset_ret

            # 更新投资组合价值
            portfolio *= (asset_ret + 1)

            # 记录当前权重
            current_weights = self.calculate_weights(portfolio)
            self.daily_weights.loc[date] = pd.Series(current_weights)

            # 检查是否需要重新平衡
            portfolio, rebalanced, reason = self.rebalance_portfolio(portfolio, current_weights, date)

            # 记录调仓标志
            self.rebalance_flags.loc[date] = rebalanced

            # 记录投资组合价值
            self.portfolio_values.loc[date] = portfolio.sum()

            # 计算投资组合日收益率
            self.daily_returns.loc[date] = (self.portfolio_values.loc[date] /
                                            self.portfolio_values.loc[prev_date] - 1)

        return self.daily_returns.dropna()


class StopLossBacktestResult(BacktestResult):
    def __init__(self, returns, weights, stop_loss_threshold=0.03, N_day=15, max_threshold=0.1, min_threshold=0.1):
        super().__init__(returns, weights, max_threshold, min_threshold)
        self.stop_loss_threshold = stop_loss_threshold
        self.N_day = N_day
        self.stop_loss_status = pd.Series(False, index=self.dates)  # 记录止损状态

    def rebalance_portfolio(self, portfolio, current_weights, date):
        """重新平衡投资组合，考虑止损状态"""
        # 检查止损状态
        if self.stop_loss_status[date]:
            # 如果处于止损状态，不需要根据阈值调仓
            return portfolio, False, "处于止损状态"
        else:
            # 正常情况下的调仓逻辑
            return super().rebalance_portfolio(portfolio, current_weights, date)

    def get_next_day_weights(self, current_weights, date):
        """根据当前权重、目标权重和止损状态，判断下一日是否需要调仓"""
        # 检查是否处于止损状态
        if self.stop_loss_status[date]:
            # 如果处于止损状态，下一日保持止损状态
            next_weights = current_weights.copy()
            return next_weights, False, "处于止损状态，暂不调仓"
        else:
            # 否则使用父类的调仓逻辑
            return super().get_next_day_weights(current_weights)

    def run_backtest(self):
        """运行带止损的回测"""
        # 初始化
        self.portfolio_values.iloc[0] = 1
        portfolio = self.portfolio_values.iloc[0] * pd.Series(self.weights)
        self.daily_weights.iloc[0] = pd.Series(self.weights)
        self.rebalance_flags.iloc[0] = True  # 初始建仓视为调仓
        self.rebalance_dates.append(self.dates[0])
        self.asset_returns.iloc[0] = 0

        # 计算股票N日累计收益率
        stock_return = self.returns['Stock']
        stock_cum_return = ((stock_return + 1).rolling(window=self.N_day).apply(np.prod, raw=True) - 1).fillna(0)

        # 逐日回测
        for i in range(1, len(self.dates)):
            date = self.dates[i]
            prev_date = self.dates[i - 1]

            # 检查是否触发止损
            if stock_cum_return[date] < -self.stop_loss_threshold:
                self.stop_loss_status[date] = True

                # 如果是第一次触发止损，设置调仓标志
                if not self.stop_loss_status[prev_date]:
                    self.rebalance_flags[date] = True
                    self.rebalance_dates.append(date)
            else:
                self.stop_loss_status[date] = False

            # 计算当日各资产收益
            asset_ret = self.returns.loc[date]
            self.asset_returns.loc[date] = asset_ret

            # 更新投资组合价值
            portfolio *= (asset_ret + 1)

            # 如果处于止损状态，调整权重（卖出股票）
            if self.stop_loss_status[date]:
                # 计算总价值
                total_value = portfolio.sum()

                # 将股票权重设为0，债券和黄金按原目标权重比例分配
                bond_weight_target = self.weights['Bond'] / (self.weights['Bond'] + self.weights['Gold'])
                gold_weight_target = self.weights['Gold'] / (self.weights['Bond'] + self.weights['Gold'])

                portfolio['Stock'] = 0
                portfolio['Bond'] = bond_weight_target * total_value
                portfolio['Gold'] = gold_weight_target * total_value

                current_weights = {'Stock': 0, 'Bond': bond_weight_target, 'Gold': gold_weight_target}
            else:
                # 记录当前权重
                current_weights = self.calculate_weights(portfolio)

            self.daily_weights.loc[date] = pd.Series(current_weights)

            # 检查是否需要重新平衡（非止损状态下的正常调仓）
            if not self.stop_loss_status[date]:
                portfolio, rebalanced, reason = self.rebalance_portfolio(portfolio, current_weights, date)
                self.rebalance_flags.loc[date] = rebalanced
                if rebalanced:
                    self.rebalance_dates.append(date)

            # 记录投资组合价值
            self.portfolio_values.loc[date] = portfolio.sum()

            # 计算投资组合日收益率
            self.daily_returns.loc[date] = (self.portfolio_values.loc[date] /
                                            self.portfolio_values.loc[prev_date] - 1)

        return self.daily_returns.dropna()


def calculate_performance_metrics(strategy_returns, benchmark_returns=None, days_per_year=252):
    """计算性能指标"""
    if len(strategy_returns) == 0:
        return {}

    # 计算净值曲线
    strategy_nav = (1 + strategy_returns).cumprod()

    # 计算总收益率
    total_return = strategy_nav.iloc[-1] - 1

    # 计算年化收益率
    years = len(strategy_returns) / days_per_year
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    # 计算年化波动率
    annualized_volatility = strategy_returns.std() * np.sqrt(days_per_year)

    # 计算夏普比率（假设无风险利率为0）
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0

    # 计算最大回撤
    rolling_max = strategy_nav.cummax()
    drawdown = (strategy_nav - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # 找到最大回撤的开始和结束日期
    drawdown_duration = drawdown[drawdown == 0].index
    if len(drawdown_duration) > 0:
        max_dd_start = rolling_max[drawdown == max_drawdown].index[0]
        max_dd_end = drawdown[drawdown == max_drawdown].index[0]
        max_dd_period = f"{max_dd_start.date()} to {max_dd_end.date()} ({len(drawdown.loc[max_dd_start:max_dd_end])} days)"
    else:
        max_dd_period = "N/A"

    # 计算胜率（正收益天数比例）
    win_rate = (strategy_returns > 0).sum() / len(strategy_returns)

    # 计算盈利/亏损比率
    winning_returns = strategy_returns[strategy_returns > 0]
    losing_returns = strategy_returns[strategy_returns < 0]
    if len(winning_returns) > 0 and len(losing_returns) > 0:
        avg_win = winning_returns.mean()
        avg_loss = abs(losing_returns.mean())
        profit_loss_ratio = avg_win / avg_loss if avg_loss != 0 else 0
    else:
        profit_loss_ratio = 0

    # 初始化基准相关指标
    benchmark_total_return = None
    information_ratio = None

    # 如果提供了基准收益率，计算基准指标和信息比率
    if benchmark_returns is not None:
        # 对齐基准收益率的时间索引
        aligned_benchmark = benchmark_returns.reindex(strategy_returns.index).dropna()

        if len(aligned_benchmark) > 0:
            # 计算基准总收益率
            benchmark_nav = (1 + aligned_benchmark).cumprod()
            benchmark_total_return = benchmark_nav.iloc[-1] - 1

            # 计算超额收益率（策略收益率 - 基准收益率）
            excess_returns = strategy_returns - aligned_benchmark

            # 计算信息比率（超额收益的年化均值 / 超额收益的年化标准差）
            excess_return_mean = excess_returns.mean() * days_per_year
            excess_return_std = excess_returns.std() * np.sqrt(days_per_year)
            information_ratio = excess_return_mean / excess_return_std if excess_return_std != 0 else 0

    return {
        'Annualized Return': annualized_return,
        'Annualized Volatility': annualized_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Max Drawdown': max_drawdown,
        'Max Drawdown Period': max_dd_period,
        'Win Rate': win_rate,
        'Profit/Loss Ratio': profit_loss_ratio
    }


def plot_nav_returns(daily_returns, benchmark_returns, title="策略净值与收益率", figsize=(12, 8)):
    """绘制净值曲线和收益率分布图"""
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    # 计算净值
    strategy_nav = (1 + daily_returns).cumprod()
    benchmark_nav = (1 + benchmark_returns).cumprod()

    # 子图1：净值曲线
    axes[0].plot(strategy_nav.index, strategy_nav.values, label='策略净值', linewidth=2, color='blue')
    axes[0].plot(benchmark_nav.index, benchmark_nav.values, label='基准净值', linewidth=2, color='red', alpha=0.7)
    axes[0].set_title(f'{title} - 净值曲线', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('日期', fontsize=12)
    axes[0].set_ylabel('净值（从1开始）', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='best')

    # 添加最终净值信息
    final_strategy = strategy_nav.iloc[-1]
    final_benchmark = benchmark_nav.iloc[-1]
    axes[0].text(0.02, 0.02,
                 f'策略最终净值: {final_strategy:.2f} | 基准最终净值: {final_benchmark:.2f}',
                 transform=axes[0].transAxes, fontsize=10,
                 bbox=dict(facecolor='yellow', alpha=0.2))

    # 子图2：收益率分布
    axes[1].hist(daily_returns.values * 100, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[1].axvline(x=daily_returns.mean() * 100, color='red', linestyle='--', linewidth=2,
                    label=f'均值: {daily_returns.mean() * 100:.2f}%')
    axes[1].axvline(x=0, color='green', linestyle='-', linewidth=1, label='零收益线')
    axes[1].set_title(f'{title} - 收益率分布', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('日收益率 (%)', fontsize=12)
    axes[1].set_ylabel('频率', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='best')

    plt.tight_layout()
    return fig


def print_table(data, title):
    """打印表格"""
    print(f"\n{title}")
    print("=" * 60)

    if isinstance(data, pd.DataFrame):
        # 格式化DataFrame显示
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        print(data.to_string())
    elif isinstance(data, dict):
        # 格式化字典显示
        for key, value in data.items():
            if isinstance(value, float):
                if key in ['Win Rate', 'Profit/Loss Ratio']:
                    print(f"{key}: {value:.3f}")
                elif key in ['Annualized Return', 'Annualized Volatility', 'Max Drawdown', 'Sharpe Ratio']:
                    print(f"{key}: {value:.3f}")
                else:
                    print(f"{key}: {value:.3f}")
            else:
                print(f"{key}: {value}")
    print("=" * 60)


def generate_html_report(result_no_stop, result_stop, benchmark_returns, last_date):
    """生成HTML格式的报告"""

    # 获取最新持仓和表现
    latest_date = result_no_stop.dates[-1]
    latest_weights_no_stop = result_no_stop.daily_weights.loc[latest_date]
    latest_weights_stop = result_stop.daily_weights.loc[latest_date]

    # 获取下一日持仓建议
    next_weights_no_stop, needs_rebalance_no_stop, reason_no_stop = result_no_stop.get_next_day_weights(
        latest_weights_no_stop)
    next_weights_stop, needs_rebalance_stop, reason_stop = result_stop.get_next_day_weights(latest_weights_stop,
                                                                                            latest_date)

    # 今日策略收益率
    today_strategy_return = pd.DataFrame({
        'ret_all': [result_no_stop.daily_returns.loc[latest_date]],
        'ret_long': [result_no_stop.daily_returns.loc[latest_date]],
        'ret_short': [result_stop.daily_returns.loc[latest_date]]
    }, index=[latest_date])

    # 今日持仓
    holdings_data = []
    for asset in ['Stock', 'Bond', 'Gold']:
        holdings_data.append({
            'symbol': ASSET_NAMES[ASSET_CODES[asset]],
            'side': 'long',
            'weight': latest_weights_no_stop[asset],
            'return': returns.loc[latest_date, asset]
        })

    today_holdings = pd.DataFrame(holdings_data)

    # 下一日持仓建议 - 无止损策略
    next_day_holdings_no_stop_data = []
    for asset in ['Stock', 'Bond', 'Gold']:
        next_day_holdings_no_stop_data.append({
            '资产类别': ASSET_NAMES[ASSET_CODES[asset]],
            '当前权重': f"{latest_weights_no_stop[asset]:.1%}",
            '建议权重': f"{next_weights_no_stop[asset]:.1%}",
            '目标权重': f"{weights[asset]:.0%}",
            '变化': f"{(next_weights_no_stop[asset] - latest_weights_no_stop[asset]):+.1%}" if needs_rebalance_no_stop else "无变化"
        })

    next_day_holdings_no_stop = pd.DataFrame(next_day_holdings_no_stop_data)

    # 下一日持仓建议 - 止损策略
    next_day_holdings_stop_data = []
    for asset in ['Stock', 'Bond', 'Gold']:
        next_day_holdings_stop_data.append({
            '资产类别': ASSET_NAMES[ASSET_CODES[asset]],
            '当前权重': f"{latest_weights_stop[asset]:.1%}",
            '建议权重': f"{next_weights_stop[asset]:.1%}",
            '目标权重': f"{weights[asset]:.0%}",
            '变化': f"{(next_weights_stop[asset] - latest_weights_stop[asset]):+.1%}" if needs_rebalance_stop else "无变化"
        })

    next_day_holdings_stop = pd.DataFrame(next_day_holdings_stop_data)

    # 过去一个月指标
    month_returns_no_stop = result_no_stop.daily_returns.tail(22)
    month_returns_stop = result_stop.daily_returns.tail(22)

    month_metrics_no_stop = calculate_performance_metrics(month_returns_no_stop, benchmark_returns.tail(22))
    month_metrics_stop = calculate_performance_metrics(month_returns_stop, benchmark_returns.tail(22))

    # 创建过去一个月指标表格
    month_metrics_df = pd.DataFrame({
        'All Trades': [month_metrics_no_stop['Annualized Return'],
                       month_metrics_no_stop['Annualized Volatility'],
                       month_metrics_no_stop['Sharpe Ratio'],
                       month_metrics_no_stop['Max Drawdown'],
                       month_metrics_no_stop['Max Drawdown Period'],
                       month_metrics_no_stop['Win Rate'],
                       month_metrics_no_stop['Profit/Loss Ratio']],
        'Long Trades': [month_metrics_no_stop['Annualized Return'],
                        month_metrics_no_stop['Annualized Volatility'],
                        month_metrics_no_stop['Sharpe Ratio'],
                        month_metrics_no_stop['Max Drawdown'],
                        month_metrics_no_stop['Max Drawdown Period'],
                        month_metrics_no_stop['Win Rate'],
                        month_metrics_no_stop['Profit/Loss Ratio']],
        'Short Trades': [month_metrics_stop['Annualized Return'],
                         month_metrics_stop['Annualized Volatility'],
                         month_metrics_stop['Sharpe Ratio'],
                         month_metrics_stop['Max Drawdown'],
                         month_metrics_stop['Max Drawdown Period'],
                         month_metrics_stop['Win Rate'],
                         month_metrics_stop['Profit/Loss Ratio']]
    }, index=['Annualized Return', 'Annualized Volatility', 'Sharpe Ratio',
              'Max Drawdown', 'Max Drawdown Period', 'Win Rate', 'Profit/Loss Ratio'])

    # 过去一年指标
    year_returns_no_stop = result_no_stop.daily_returns.tail(252)
    year_returns_stop = result_stop.daily_returns.tail(252)

    year_metrics_no_stop = calculate_performance_metrics(year_returns_no_stop, benchmark_returns.tail(252))
    year_metrics_stop = calculate_performance_metrics(year_returns_stop, benchmark_returns.tail(252))

    # 创建过去一年指标表格
    year_metrics_df = pd.DataFrame({
        'All Trades': [year_metrics_no_stop['Annualized Return'],
                       year_metrics_no_stop['Annualized Volatility'],
                       year_metrics_no_stop['Sharpe Ratio'],
                       year_metrics_no_stop['Max Drawdown'],
                       year_metrics_no_stop['Max Drawdown Period'],
                       year_metrics_no_stop['Win Rate'],
                       year_metrics_no_stop['Profit/Loss Ratio']],
        'Long Trades': [year_metrics_no_stop['Annualized Return'],
                        year_metrics_no_stop['Annualized Volatility'],
                        year_metrics_no_stop['Sharpe Ratio'],
                        year_metrics_no_stop['Max Drawdown'],
                        year_metrics_no_stop['Max Drawdown Period'],
                        year_metrics_no_stop['Win Rate'],
                        year_metrics_no_stop['Profit/Loss Ratio']],
        'Short Trades': [year_metrics_stop['Annualized Return'],
                         year_metrics_stop['Annualized Volatility'],
                         year_metrics_stop['Sharpe Ratio'],
                         year_metrics_stop['Max Drawdown'],
                         year_metrics_stop['Max Drawdown Period'],
                         year_metrics_stop['Win Rate'],
                         year_metrics_stop['Profit/Loss Ratio']]
    }, index=['Annualized Return', 'Annualized Volatility', 'Sharpe Ratio',
              'Max Drawdown', 'Max Drawdown Period', 'Win Rate', 'Profit/Loss Ratio'])

    # 生成图表
    fig_month = plot_nav_returns(month_returns_no_stop, benchmark_returns.tail(22), "过去一个月策略表现")
    fig_year = plot_nav_returns(year_returns_no_stop, benchmark_returns.tail(252), "过去一年策略表现")

    # 将图表转换为base64字符串
    def fig_to_base64(fig):
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

    month_chart = fig_to_base64(fig_month)
    year_chart = fig_to_base64(fig_year)

    # 生成HTML
    html_content = f"""
<html>
    <head>
        <title>Strategy Dashboard - Last Date: {last_date}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            h1 {{ border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
            h2 {{ color: #4CAF50; margin-top: 30px; border-left: 4px solid #4CAF50; padding-left: 10px; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 15px; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #f2f2f2; font-weight: bold; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .rebalance {{ background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 15px 0; }}
            .no-rebalance {{ background-color: #d4edda; border-left: 4px solid #28a745; padding: 15px; margin: 15px 0; }}
            .highlight {{ background-color: #e7f3fe; border-left: 4px solid #2196F3; padding: 15px; margin: 15px 0; }}
            .footer {{ margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666; font-size: 12px; }}
            .target-weights {{ margin: 20px 0; padding: 15px; background-color: #f8f9fa; border-radius: 5px; }}
            .target-weights ul {{ margin: 10px 0; }}
        </style>
    </head>
    <body>
        <h1>策略监控面板 - 报告日期: {last_date}</h1>

        <div class="target-weights">
            <h3>目标资产配置权重</h3>
            <ul>
                <li>股票 (沪深300): {weights['Stock']:.0%}</li>
                <li>债券 (国债ETF): {weights['Bond']:.0%}</li>
                <li>黄金 (黄金ETF): {weights['Gold']:.0%}</li>
            </ul>
            <p>调仓阈值: ±{max_threshold:.0%} (当资产权重偏离目标权重超过阈值时调仓)</p>
        </div>

        <h2>今日持仓情况</h2>
        {today_holdings.to_html(index=False)}

        <h2>下一交易日持仓建议 - 无止损策略</h2>
        {"<div class='rebalance'><strong>📈 需要调仓:</strong> " + reason_no_stop + "</div>" if needs_rebalance_no_stop else "<div class='no-rebalance'><strong>✅ 不需要调仓:</strong> " + reason_no_stop + "</div>"}
        {next_day_holdings_no_stop.to_html(index=False)}

        <h2>下一交易日持仓建议 - 带止损策略</h2>
        {"<div class='rebalance'><strong>📈 需要调仓:</strong> " + reason_stop + "</div>" if needs_rebalance_stop else "<div class='no-rebalance'><strong>✅ 不需要调仓:</strong> " + reason_stop + "</div>"}
        {next_day_holdings_stop.to_html(index=False)}

        <div class="highlight">
            <h3>调仓频率统计</h3>
            <p>无止损策略总调仓次数: {len(result_no_stop.rebalance_dates)} 次</p>
            <p>带止损策略总调仓次数: {len(result_stop.rebalance_dates)} 次</p>
            <p>最近一次调仓日期: {result_no_stop.rebalance_dates[-1] if result_no_stop.rebalance_dates else '无'}</p>
            <p>注: 调仓频率较低，通常只在资产权重偏离目标超过阈值时进行调仓。</p>
        </div>

        <h2>今日策略收益率</h2>
        {today_strategy_return.to_html()}

        <h2>过去一个月策略表现</h2>
        {month_metrics_df.to_html()}
        <img src="data:image/png;base64,{month_chart}" style="width:100%; max-width:1400px; display:block; margin:20px auto;">

        <h2>过去一年策略表现</h2>
        {year_metrics_df.to_html()}
        <img src="data:image/png;base64,{year_chart}" style="width:100%; max-width:1400px; display:block; margin:20px auto;">

        <div class="footer">
            <p>报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>数据时间范围: {returns.index[0].date()} 至 {returns.index[-1].date()} (共{len(returns)}个交易日)</p>
            <p>免责声明: 本报告仅供参考，不构成投资建议。投资有风险，入市需谨慎。</p>
        </div>
    </body>
</html>
"""

    # 保存HTML文件
    html_filename = "harry_browne_strategy_dashboard.html"
    with open(html_filename, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"\nHTML报告已保存到 '{html_filename}'")

    return html_content, html_filename


def _build_attachment(file_path: str) -> dict:
    """构建邮件附件"""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Attachment not found: {file_path}")

    with open(path, "rb") as f:
        file_data = f.read()
    encoded_file = base64.b64encode(file_data).decode()

    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "application/octet-stream"

    return {
        "content": encoded_file,
        "filename": path.name,
        "type": mime_type,
        "disposition": "attachment"
    }


def send_email_with_attachment(addresses, subject: str, html_body: str = None, text_body: str = None,
                               file_path: str = None):
    """
    发送邮件（使用SendGrid API）

    参数:
        addresses: 收件人邮箱列表或单个邮箱地址
        subject: 邮件主题
        html_body: HTML邮件正文（可选）
        text_body: 纯文本邮件正文（可选）
        file_path: 附件文件路径（可选）
    """
    if not SENDGRID_API_KEY or not SENDER_EMAIL:
        print("错误: 请先配置SendGrid API密钥和发件人邮箱")
        return

    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition
    except ImportError:
        print("错误: 请先安装sendgrid库，使用命令: pip install sendgrid")
        return

    # 准备收件人列表
    if isinstance(addresses, str):
        to_emails = [addresses]
    else:
        to_emails = list(addresses)

    if not to_emails:
        print("错误: 没有指定收件人邮箱")
        return

    # 准备邮件正文
    if not html_body and not text_body:
        html_body = "<p>Please find the attached file.</p>"

    # 创建邮件对象
    message = Mail(
        from_email=SENDER_EMAIL,
        to_emails=to_emails,
        subject=subject,
        html_content=html_body if html_body else None,
        plain_text_content=text_body if text_body else None
    )

    # 添加附件
    if file_path:
        attachment = _build_attachment(file_path)
        message.attachment = Attachment(
            FileContent(attachment["content"]),
            FileName(attachment["filename"]),
            FileType(attachment["type"]),
            Disposition(attachment["disposition"])
        )

    try:
        # 发送邮件
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)

        print(f"邮件发送成功!")
        print(f"状态码: {response.status_code}")
        print(f"收件人: {', '.join(to_emails)}")
        print(f"主题: {subject}")
        if file_path:
            print(f"附件: {file_path}")

    except Exception as e:
        print(f"邮件发送失败: {e}")


def create_email_content(result_no_stop, result_stop, latest_date):
    """创建邮件正文内容"""

    # 获取最新数据
    latest_weights_no_stop = result_no_stop.daily_weights.loc[latest_date]
    latest_weights_stop = result_stop.daily_weights.loc[latest_date]

    # 获取下一日持仓建议
    next_weights_no_stop, needs_rebalance_no_stop, reason_no_stop = result_no_stop.get_next_day_weights(
        latest_weights_no_stop)
    next_weights_stop, needs_rebalance_stop, reason_stop = result_stop.get_next_day_weights(latest_weights_stop,
                                                                                            latest_date)

    # 今日收益率
    today_return_no_stop = result_no_stop.daily_returns.loc[latest_date]
    today_return_stop = result_stop.daily_returns.loc[latest_date]
    today_benchmark_return = benchmark_returns.loc[latest_date]

    # 创建HTML邮件正文
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>资产配置策略日报 - {latest_date.date()}</title>
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 1000px; margin: 0 auto; padding: 20px; }}
        .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; margin-bottom: 30px; text-align: center; }}
        .section {{ margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
        .section-title {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; margin-bottom: 20px; }}
        table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; font-weight: bold; }}
        .positive {{ color: #27ae60; font-weight: bold; }}
        .negative {{ color: #e74c3c; font-weight: bold; }}
        .highlight {{ background-color: #fffacd; padding: 10px; border-radius: 5px; margin: 10px 0; }}
        .rebalance {{ background-color: #ffeaa7; padding: 15px; border-left: 4px solid #fdcb6e; margin: 15px 0; }}
        .no-rebalance {{ background-color: #d1f7c4; padding: 15px; border-left: 4px solid #55efc4; margin: 15px 0; }}
        .footer {{ margin-top: 30px; padding: 20px; background-color: #f4f4f4; border-radius: 5px; text-align: center; font-size: 12px; color: #666; }}
        .info-box {{ background-color: #e3f2fd; padding: 15px; border-radius: 5px; margin: 15px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 资产配置策略日报</h1>
        <h2>{latest_date.date()} 日度报告</h2>
        <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="info-box">
        <h3>📋 报告摘要</h3>
        <p>本报告包含今日表现、当前持仓和下一交易日调仓建议。</p>
        <p><strong>目标配置:</strong> 股票 {weights['Stock']:.0%} | 债券 {weights['Bond']:.0%} | 黄金 {weights['Gold']:.0%}</p>
        <p><strong>调仓阈值:</strong> ±{max_threshold:.0%} (当权重偏离目标超过阈值时调仓)</p>
    </div>

    <div class="section">
        <h2 class="section-title">📈 今日表现摘要</h2>
        <table>
            <tr>
                <th>策略类型</th>
                <th>今日收益率</th>
                <th>状态</th>
            </tr>
            <tr>
                <td>无止损策略</td>
                <td class="{'positive' if today_return_no_stop > 0 else 'negative'}">{today_return_no_stop:.2%}</td>
                <td><span class="{'positive' if today_return_no_stop > 0 else 'negative'}">{"盈利" if today_return_no_stop > 0 else "亏损"}</span></td>
            </tr>
            <tr>
                <td>带止损策略</td>
                <td class="{'positive' if today_return_stop > 0 else 'negative'}">{today_return_stop:.2%}</td>
                <td><span class="{'positive' if today_return_stop > 0 else 'negative'}">{"盈利" if today_return_stop > 0 else "亏损"}</span></td>
            </tr>
            <tr>
                <td>基准 (沪深300)</td>
                <td class="{'positive' if today_benchmark_return > 0 else 'negative'}">{today_benchmark_return:.2%}</td>
                <td>-</td>
            </tr>
        </table>
    </div>

    <div class="section">
        <h2 class="section-title">🏦 今日持仓情况</h2>
        <h3>无止损策略当前持仓:</h3>
        <table>
            <tr>
                <th>资产类别</th>
                <th>持仓比例</th>
                <th>目标比例</th>
                <th>今日收益率</th>
                <th>偏离程度</th>
            </tr>
"""

    # 添加无止损策略持仓信息
    for asset in ['Stock', 'Bond', 'Gold']:
        asset_name = ASSET_NAMES.get(ASSET_CODES[asset], asset)
        asset_return = returns.loc[latest_date, asset]
        deviation = latest_weights_no_stop[asset] - weights[asset]
        html_content += f"""
            <tr>
                <td>{asset_name}</td>
                <td>{latest_weights_no_stop[asset]:.2%}</td>
                <td>{weights[asset]:.0%}</td>
                <td class="{'positive' if asset_return > 0 else 'negative'}">{asset_return:.2%}</td>
                <td class="{'positive' if deviation <= 0 else 'negative'}">{deviation:+.2%}</td>
            </tr>
"""

    html_content += """
        </table>

        <h3>带止损策略当前持仓:</h3>
        <table>
            <tr>
                <th>资产类别</th>
                <th>持仓比例</th>
                <th>目标比例</th>
                <th>今日收益率</th>
                <th>偏离程度</th>
            </tr>
"""

    # 添加带止损策略持仓信息
    for asset in ['Stock', 'Bond', 'Gold']:
        asset_name = ASSET_NAMES.get(ASSET_CODES[asset], asset)
        asset_return = returns.loc[latest_date, asset]
        deviation = latest_weights_stop[asset] - weights[asset]
        html_content += f"""
            <tr>
                <td>{asset_name}</td>
                <td>{latest_weights_stop[asset]:.2%}</td>
                <td>{weights[asset]:.0%}</td>
                <td class="{'positive' if asset_return > 0 else 'negative'}">{asset_return:.2%}</td>
                <td class="{'positive' if deviation <= 0 else 'negative'}">{deviation:+.2%}</td>
            </tr>
"""

    html_content += f"""
        </table>
    </div>

    <div class="section">
        <h2 class="section-title">🔄 下一交易日调仓建议</h2>

        <h3>无止损策略:</h3>
        {"<div class='rebalance'><strong>📈 需要调仓:</strong> " + reason_no_stop + "</div>" if needs_rebalance_no_stop else "<div class='no-rebalance'><strong>✅ 不需要调仓:</strong> " + reason_no_stop + "</div>"}
        <table>
            <tr>
                <th>资产类别</th>
                <th>当前权重</th>
                <th>建议权重</th>
                <th>变化</th>
            </tr>
"""

    # 添加无止损策略调仓建议
    for asset in ['Stock', 'Bond', 'Gold']:
        asset_name = ASSET_NAMES.get(ASSET_CODES[asset], asset)
        change = next_weights_no_stop[asset] - latest_weights_no_stop[asset]
        html_content += f"""
            <tr>
                <td>{asset_name}</td>
                <td>{latest_weights_no_stop[asset]:.1%}</td>
                <td>{next_weights_no_stop[asset]:.1%}</td>
                <td class="{'positive' if change >= 0 else 'negative'}">{change:+.1%}</td>
            </tr>
"""

    html_content += f"""
        </table>

        <h3>带止损策略:</h3>
        {"<div class='rebalance'><strong>📈 需要调仓:</strong> " + reason_stop + "</div>" if needs_rebalance_stop else "<div class='no-rebalance'><strong>✅ 不需要调仓:</strong> " + reason_stop + "</div>"}
        <table>
            <tr>
                <th>资产类别</th>
                <th>当前权重</th>
                <th>建议权重</th>
                <th>变化</th>
            </tr>
"""

    # 添加带止损策略调仓建议
    for asset in ['Stock', 'Bond', 'Gold']:
        asset_name = ASSET_NAMES.get(ASSET_CODES[asset], asset)
        change = next_weights_stop[asset] - latest_weights_stop[asset]
        html_content += f"""
            <tr>
                <td>{asset_name}</td>
                <td>{latest_weights_stop[asset]:.1%}</td>
                <td>{next_weights_stop[asset]:.1%}</td>
                <td class="{'positive' if change >= 0 else 'negative'}">{change:+.1%}</td>
            </tr>
"""

    html_content += f"""
        </table>

        <div class="highlight">
            <p><strong>📊 调仓频率统计:</strong></p>
            <p>• 无止损策略总调仓次数: {len(result_no_stop.rebalance_dates)} 次</p>
            <p>• 带止损策略总调仓次数: {len(result_stop.rebalance_dates)} 次</p>
            <p>• 最近一次调仓: {result_no_stop.rebalance_dates[-1].date() if result_no_stop.rebalance_dates else '无'}</p>
            <p><em>注: 由于使用阈值调仓机制，调仓频率较低，通常只在资产权重偏离目标超过±{max_threshold:.0%}时才进行调仓。</em></p>
        </div>
    </div>

    <div class="section">
        <h2 class="section-title">📊 报告说明</h2>
        <p>本报告包含以下内容:</p>
        <ul>
            <li><strong>今日表现摘要</strong>: 展示两种策略和基准的今日收益率</li>
            <li><strong>今日持仓情况</strong>: 展示当前各资产的持仓比例和偏离程度</li>
            <li><strong>下一交易日调仓建议</strong>: 基于阈值规则判断是否需要调仓及调仓建议</li>
            <li><strong>详细分析报告</strong>: 查看附件中的完整HTML报告，包含:
                <ul>
                    <li>过去一个月和一年的详细绩效指标</li>
                    <li>净值曲线和收益率分布图表</li>
                    <li>完整的策略对比分析</li>
                </ul>
            </li>
        </ul>
        <p><strong>重要提示</strong>: 投资有风险，本报告仅供参考，不构成投资建议。实际投资请谨慎决策。</p>
    </div>

    <div class="footer">
        <p>📧 本邮件由资产配置策略系统自动生成</p>
        <p>🔄 如需取消订阅或调整报告频率，请联系系统管理员</p>
        <p>🔒 本邮件包含机密信息，仅供指定收件人使用</p>
    </div>
</body>
</html>
"""

    # 创建纯文本邮件正文（备用）
    text_content = f"""
资产配置策略日报 - {latest_date.date()}

今日表现摘要:
- 无止损策略: {today_return_no_stop:.2%}
- 带止损策略: {today_return_stop:.2%}
- 基准 (沪深300): {today_benchmark_return:.2%}

今日持仓情况 (无止损策略):
"""

    for asset in ['Stock', 'Bond', 'Gold']:
        asset_name = ASSET_NAMES.get(ASSET_CODES[asset], asset)
        deviation = latest_weights_no_stop[asset] - weights[asset]
        text_content += f"- {asset_name}: {latest_weights_no_stop[asset]:.2%} (目标: {weights[asset]:.0%}, 偏离: {deviation:+.2%}, 今日: {returns.loc[latest_date, asset]:.2%})\n"

    text_content += f"\n下一交易日调仓建议 (无止损策略):\n"

    if needs_rebalance_no_stop:
        text_content += f"需要调仓: {reason_no_stop}\n"
        for asset in ['Stock', 'Bond', 'Gold']:
            asset_name = ASSET_NAMES.get(ASSET_CODES[asset], asset)
            change = next_weights_no_stop[asset] - latest_weights_no_stop[asset]
            text_content += f"- {asset_name}: {latest_weights_no_stop[asset]:.1%} → {next_weights_no_stop[asset]:.1%} ({change:+.1%})\n"
    else:
        text_content += f"不需要调仓: {reason_no_stop}\n"

    text_content += f"\n调仓频率统计:\n"
    text_content += f"- 无止损策略总调仓次数: {len(result_no_stop.rebalance_dates)} 次\n"
    text_content += f"- 最近一次调仓: {result_no_stop.rebalance_dates[-1].date() if result_no_stop.rebalance_dates else '无'}\n"

    text_content += "\n请查看附件中的完整HTML报告获取详细分析。\n\n本邮件由资产配置策略系统自动生成。"

    return html_content, text_content


# 运行回测
print("=" * 70)
print("开始运行资产配置策略回测")
print("=" * 70)

# 运行无止损策略回测
print("\n1. 运行无止损策略回测...")
result_no_stop = BacktestResult(returns, weights, max_threshold, min_threshold)
daily_returns_no_stop = result_no_stop.run_backtest()

# 运行带止损策略回测
print("2. 运行带止损策略回测...")
result_stop = StopLossBacktestResult(returns, weights, stop_loss_threshold=0.03, N_day=15)
daily_returns_stop = result_stop.run_backtest()

# 基准收益率（沪深300）
benchmark_returns = returns['Stock']

# 获取最新数据
latest_date = returns.index[-1]

print("\n" + "=" * 70)
print("策略回测结果报告")
print("=" * 70)
print(f"报告日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"数据时间范围: {returns.index[0].date()} 至 {returns.index[-1].date()}")
print(f"总交易日数: {len(returns)}")

# 生成HTML报告
print("\n" + "=" * 70)
print("生成HTML报告...")
print("=" * 70)

html_content, html_filename = generate_html_report(result_no_stop, result_stop, benchmark_returns, latest_date.date())

# 发送邮件
print("\n" + "=" * 70)
print("发送邮件...")
print("=" * 70)

# send emails
HTML_PATH = "harry_browne_strategy_dashboard.html"
try:
    with open(HTML_PATH, "r", encoding="utf-8") as f:
        HTML_BODY = f.read()
except Exception:
    HTML_BODY = "<p>Please find the attached file.</p>"
# HTML_BODY = "<p>Please find the attached file.</p>"
for re in pw.RECIPIENTS:
    print(f"Sending {HTML_PATH} to {re}")
    email_sender_v2.send_html_email_with_attachment(
        smtp_server="smtp.gmail.com",
        smtp_port=587,
        sender_email=pw.SENDER_EMAIL,    # your gmail
        password=pw.google_email_app_password,  # your gmail app password
        receiver_email=re,  # recipient email
        subject="Harry Brown Daily Backtest Report",
        html_body=HTML_BODY,
        attachment_path=HTML_PATH
    )

print("\n" + "=" * 70)
print("回测完成！")
print("=" * 70)
print("总结:")
print(f"1. 回测数据: {len(returns)} 个交易日")
print(f"2. HTML报告: 已生成并保存为 '{html_filename}'")
print(f"3. 邮件发送: 已发送给 {len(pw.RECIPIENTS)} 个收件人")
print(f"4. 报告日期: {latest_date.date()}")
print(f"5. 无止损策略调仓次数: {len(result_no_stop.rebalance_dates)} 次")
print(f"6. 带止损策略调仓次数: {len(result_stop.rebalance_dates)} 次")
print("\n所有任务已完成!")