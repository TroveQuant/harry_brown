"""
双低可转债策略每日运行
"""

import tushare as ts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm
import base64
from io import BytesIO
import argparse
import os
from pathlib import Path
import email_sender_v2
import a_passwards as pw

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ==================== 配置参数 ====================
TUSHARE_TOKEN = pw.TUSHARE_TOKEN  # 请替换为您的有效Token

# 策略参数
X_WEIGHT = 1.6          # 双低因子中溢价率的权重
TOP_N = 10               # 每周持有可转债数量

# 止盈止损参数
STOP_GAIN = 0.03         # 止盈回撤阈值
STOP_LOSS = -0.05        # 止损阈值
REENTRY_THRESHOLD = 0.01 # 再入场反弹阈值

# 哈利布朗永久组合权重
HARRY_WEIGHTS = {
    'Stock': 0.25,
    'Bond': 0.25,
    'Gold': 0.25,
    'Cash': 0.25
}

# 资产配置（全部使用Tushare）
ASSET_CONFIG = {
    'Stock': {'code': '000300.SH', 'type': 'index'},   # 沪深300指数
    'Bond': {'code': '511010.SH', 'type': 'fund'},     # 国债ETF
    'Gold': {'code': '518880.SH', 'type': 'fund'},     # 黄金ETF
    'Cash': {'code': '511880.SH', 'type': 'fund'},     # 货币基金（银华日利）作为现金等价物
    'Benchmark': {'code': '000300.SH', 'type': 'index'} # 基准沪深300
}
ASSET_NAMES = {
    '000300.SH': '沪深300',
    '511010.SH': '国债ETF',
    '518880.SH': '黄金ETF',
    '511880.SH': '银华日利',
}

# 网络请求配置
MAX_RETRIES = 3
TIMEOUT = 10
POOL_CONNECTIONS = 10

# 数据过滤参数
MIN_BOND_PRICE = 80
MAX_BOND_PRICE = 130
MIN_REMAIN_SIZE = 0.5



# 数据时间范围
START_DATE = '20250101'
END_DATE = datetime.now().strftime('%Y%m%d')

# 缓存目录
CACHE_DIR = os.path.expanduser("cb_strategy_cache")
# ===================================================


class DataLoader:
    """Tushare数据加载器（支持本地缓存和增量更新，含价格插值修复）"""

    def __init__(self, token):
        self.token = token
        ts.set_token(token)

        # 带重试的session
        self.session = requests.Session()
        retry_strategy = Retry(
            total=MAX_RETRIES,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=POOL_CONNECTIONS,
            pool_maxsize=POOL_CONNECTIONS
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        self.pro = ts.pro_api()
        self._cache = {}
        self._cb_basic_cache = None
        self._stock_price_cache = {}

        # 确保缓存目录存在
        os.makedirs(CACHE_DIR, exist_ok=True)

    # ---------- 缓存管理 ----------
    def _get_cache_path(self, name):
        return os.path.join(CACHE_DIR, f"{name}.parquet")

    def _load_cb_cache(self):
        """加载本地缓存的可转债日线数据，返回DataFrame（索引为trade_date）"""
        path = self._get_cache_path("cb_daily")
        if os.path.exists(path):
            try:
                df = pd.read_parquet(path)
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df.set_index('trade_date', inplace=True)
                print('成功加载 本地缓存的可转债日线数据')
                return df
            except Exception as e:
                print(f"警告：读取可转债缓存失败，将重新获取：{e}")
                return pd.DataFrame()
        return pd.DataFrame()

    def _save_cb_cache(self, df):
        """保存可转债日线数据到缓存（覆盖）"""
        if df.empty:
            return
        path = self._get_cache_path("cb_daily")
        df_save = df.reset_index()
        df_save.to_parquet(path, index=False)
        print(f"可转债缓存已保存至 {path}")

    def _load_asset_cache(self, asset_name):
        """加载单个资产的历史价格Series，索引为日期"""
        path = self._get_cache_path(f"asset_{asset_name}")
        if os.path.exists(path):
            try:
                df = pd.read_parquet(path)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                return df['close']
            except Exception as e:
                print(f"警告：读取{asset_name}缓存失败，将重新获取：{e}")
                return pd.Series(dtype=float)
        return pd.Series(dtype=float)

    def _save_asset_cache(self, asset_name, series):
        """保存资产价格到缓存"""
        if series.empty:
            return
        df = series.reset_index()
        df.columns = ['date', 'close']
        path = self._get_cache_path(f"asset_{asset_name}")
        df.to_parquet(path, index=False)
        print(f"{asset_name}缓存已保存至 {path}")

    def _load_trade_calendar_cache(self):
        """加载本地交易日历缓存，并确保返回升序列表"""
        path = self._get_cache_path("trade_calendar")
        if os.path.exists(path):
            try:
                df = pd.read_parquet(path)
                dates = pd.to_datetime(df['cal_date']).tolist()
                return sorted(dates)
            except Exception as e:
                print(f"警告：读取交易日历缓存失败，将重新获取：{e}")
        return None

    def _save_trade_calendar_cache(self, dates):
        """保存交易日历缓存（保存前已排序）"""
        df = pd.DataFrame({'cal_date': [d.strftime('%Y%m%d') for d in dates]})
        path = self._get_cache_path("trade_calendar")
        df.to_parquet(path, index=False)

    # ---------- 数据获取方法 ----------
    def _get_cb_basic(self):
        """获取可转债基础信息（缓存）"""
        if self._cb_basic_cache is None:
            try:
                print("正在获取可转债基础信息...")
                self._cb_basic_cache = self.pro.cb_basic()
                print(f"✓ 获取到 {len(self._cb_basic_cache)} 只可转债基础信息")
            except Exception as e:
                print(f"✗ 获取可转债基础信息失败: {e}")
                self._cb_basic_cache = pd.DataFrame()
        return self._cb_basic_cache

    def get_trade_dates(self, start_date, end_date):
        """获取交易日列表（优先使用缓存）"""
        cached_dates = self._load_trade_calendar_cache()
        # if cached_dates is not None:
        #     start = pd.to_datetime(start_date)
        #     end = pd.to_datetime(end_date)
        #     filtered = [d for d in cached_dates if start <= d <= end]
        #     if filtered:
        #         return filtered
        try:
            df = self.pro.trade_cal(exchange='SSE', start_date=start_date, end_date=end_date, is_open='1')
            dates = pd.to_datetime(df['cal_date']).tolist()
            if cached_dates is not None:
                all_dates = sorted(set(cached_dates + dates))
            else:
                all_dates = sorted(dates)
            self._save_trade_calendar_cache(all_dates)
            return all_dates
        except Exception as e:
            raise ValueError(f"获取交易日历失败: {e}")

    def _batch_get_stock_prices(self, stk_codes, trade_date):
        """批量获取股票价格（线程池）"""
        if not stk_codes:
            return {}
        prices = {}
        uncached = []
        for code in stk_codes:
            key = f"{code}_{trade_date}"
            if key in self._stock_price_cache:
                prices[code] = self._stock_price_cache[key]
            else:
                uncached.append(code)
        if not uncached:
            return prices

        def fetch(code):
            try:
                df = self.pro.daily(ts_code=code, trade_date=trade_date, timeout=TIMEOUT)
                if not df.empty and 'close' in df.columns:
                    price = float(df['close'].iloc[0])
                    self._stock_price_cache[f"{code}_{trade_date}"] = price
                    return code, price
            except Exception:
                pass
            return code, None

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(fetch, code) for code in uncached]
            for future in tqdm(as_completed(futures), total=len(uncached), desc=f"获取正股价格({trade_date})", leave=False):
                code, price = future.result()
                if price is not None:
                    prices[code] = price
        return prices

    def _calculate_premium_rate(self, cb_price, conv_price, stk_price):
        """计算转股溢价率（小数）"""
        try:
            cb = float(cb_price)
            cv = float(conv_price)
            stk = float(stk_price)
            if cv <= 0 or stk <= 0:
                return None
            conversion_value = 100 / cv * stk
            return cb / conversion_value - 1
        except:
            return None

    def get_cb_daily(self, trade_date):
        """获取某日可转债行情，包含溢价率计算"""
        date_str = trade_date.strftime('%Y%m%d') if isinstance(trade_date, pd.Timestamp) else str(trade_date)

        try:
            df = self.pro.cb_daily(trade_date=date_str)
            if df is None or df.empty:
                return pd.DataFrame()

            numeric_cols = ['open', 'high', 'low', 'close', 'pre_close', 'change', 'pct_chg', 'vol', 'amount',
                            'remain_size']
            existing_numeric = [col for col in numeric_cols if col in df.columns]
            for col in existing_numeric:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            has_premium = 'conv_premium' in df.columns and not df['conv_premium'].isna().all()

            if not has_premium:
                basic = self._get_cb_basic()
                if basic.empty:
                    return pd.DataFrame()

                merge_cols = ['ts_code', 'conv_price', 'stk_code']
                available_cols = [c for c in merge_cols if c in basic.columns]

                if len(available_cols) < 2:
                    return pd.DataFrame()

                df = df.merge(basic[available_cols], on='ts_code', how='left')

                stocks_to_fetch = []
                for idx, row in df.iterrows():
                    if pd.notna(row.get('stk_code')) and pd.notna(row.get('conv_price')):
                        stocks_to_fetch.append(row['stk_code'])
                stocks_to_fetch = list(set(stocks_to_fetch))

                if stocks_to_fetch:
                    stock_prices = self._batch_get_stock_prices(stocks_to_fetch, date_str)
                else:
                    stock_prices = {}

                premium_rates = []
                for idx, row in df.iterrows():
                    conv_price = row.get('conv_price')
                    stk_code = row.get('stk_code')
                    cb_price = row.get('close')

                    if pd.notna(conv_price) and pd.notna(stk_code) and pd.notna(cb_price):
                        stk_price = stock_prices.get(stk_code)
                        if stk_price is not None:
                            premium = self._calculate_premium_rate(cb_price, conv_price, stk_price)
                            premium_rates.append(premium)
                        else:
                            premium_rates.append(None)
                    else:
                        premium_rates.append(None)

                df['conv_premium'] = premium_rates

            if 'conv_premium' not in df.columns:
                return pd.DataFrame()

            df = df.dropna(subset=['conv_premium', 'close'])
            df = df[(df['close'] >= MIN_BOND_PRICE) & (df['close'] <= MAX_BOND_PRICE)]

            if 'remain_size' in df.columns:
                df = df[df['remain_size'] >= MIN_REMAIN_SIZE]

            return df

        except Exception as e:
            print(f"✗ 获取{date_str}可转债数据失败: {e}")
            return pd.DataFrame()

    def _is_cb_date_valid(self, date, df):
        """检查某日可转债数据是否有效（至少包含必要的列且有数据）"""
        if df.empty:
            return False
        required_cols = ['ts_code', 'close', 'conv_premium']
        if not all(col in df.columns for col in required_cols):
            return False
        if df['close'].isna().all() or df['conv_premium'].isna().all():
            return False
        return True

    # ---------- Tushare获取资产数据 ----------
    def _get_tushare_asset_daily(self, name, config, start_date, end_date):
        """从Tushare获取资产日线数据"""
        try:
            if config['type'] == 'index':
                df = self.pro.index_daily(ts_code=config['code'], start_date=start_date, end_date=end_date)
            else:  # fund
                df = self.pro.fund_daily(ts_code=config['code'], start_date=start_date, end_date=end_date)

            if df is not None and not df.empty:
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df.set_index('trade_date', inplace=True)
                series = df['close'].sort_index()
                return series
            return pd.Series(dtype=float)
        except Exception as e:
            print(f"Tushare获取{name}失败: {e}")
            return pd.Series(dtype=float)

    # ---------- 价格清洗函数（插值修复） ----------
    def _clean_price_series(self, series, name, max_change=0.2, min_value=None, max_value=None):
        """
        清洗价格序列：剔除超出合理范围或单日涨跌幅过大的异常点，并用线性插值填充
        :param series: pd.Series, index为日期, values为价格
        :param name: 资产名称（用于打印警告）
        :param max_change: 允许的最大单日涨跌幅（小数），超过此值视为异常
        :param min_value: 价格合理下限（若为None则根据资产自动设置）
        :param max_value: 价格合理上限（若为None则根据资产自动设置）
        :return: 清洗后的Series
        """
        if series.empty:
            return series

        # 根据资产类型设置默认合理范围
        if min_value is None or max_value is None:
            if name in ['Stock', 'Benchmark']:  # 沪深300
                default_min = 500
                default_max = 20000
            elif name == 'Cash':                # 货币基金
                default_min = 50
                default_max = 200
            else:                                # 国债ETF、黄金ETF
                default_min = 0.1
                default_max = 200
            min_value = min_value or default_min
            max_value = max_value or default_max

        s = series.copy()
        # 1. 价格超出合理范围 → NaN
        invalid_range = (s < min_value) | (s > max_value)
        if invalid_range.any():
            print(f"警告：{name} 发现 {invalid_range.sum()} 个价格超出合理范围 [{min_value}, {max_value}]，将进行插值修复。")
            s[invalid_range] = np.nan

        # 2. 计算日收益率，标记涨跌幅过大的点（对应的价格设为NaN）
        ret = s.pct_change()
        abnormal_ret = ret.abs() > max_change
        if abnormal_ret.any():
            print(f"警告：{name} 发现 {abnormal_ret.sum()} 个异常涨跌幅（超过{max_change:.0%}），将进行插值修复。")
            abnormal_dates = ret[abnormal_ret].index
            s.loc[abnormal_dates] = np.nan

        # 3. 线性插值填充所有NaN
        if s.isna().any():
            s = s.interpolate(method='linear', limit_direction='both')
            # 若开头仍有NaN，用第一个有效值向前填充
            if pd.isna(s.iloc[0]):
                first_valid = s.first_valid_index()
                if first_valid is not None:
                    s.iloc[:s.index.get_loc(first_valid)] = s[first_valid]
            # 若结尾仍有NaN，用最后一个有效值向后填充
            if pd.isna(s.iloc[-1]):
                last_valid = s.last_valid_index()
                if last_valid is not None:
                    s.iloc[s.index.get_loc(last_valid)+1:] = s[last_valid]
        return s

    # ---------- 主数据加载函数 ----------
    def get_daily_data(self, start_date, end_date, force=False):
        """
        获取所有交易日的可转债数据及资产数据（带增量更新和价格清洗）
        :param force: 是否强制重新获取所有数据（忽略缓存）
        返回字典：{'cb_data': {date: df}, 'asset_data': {name: Series}, 'trade_dates': list}
        """
        print(f"\n加载数据: {start_date} 至 {end_date}（使用本地缓存，force={force}）")

        # 获取交易日列表
        trade_dates = self.get_trade_dates(start_date, end_date)
        trade_dates = pd.to_datetime(trade_dates)
        print(f"交易日数量: {len(trade_dates)}")

        # ---------- 可转债数据 ----------
        print("\n处理可转债数据...")
        if force:
            # 强制重新获取：删除缓存，重新获取所有交易日
            cb_cache_df = pd.DataFrame()
            missing_dates = trade_dates
            print("强制模式：将重新获取所有可转债数据")
        else:
            cb_cache_df = self._load_cb_cache()
            # 检查每个交易日的数据是否有效，而不仅仅是日期存在
            existing_dates = set(cb_cache_df.index) if not cb_cache_df.empty else set()
            missing_dates = []
            for d in trade_dates:
                if d not in existing_dates:
                    missing_dates.append(d)
                else:
                    # 日期存在，但数据可能不完整
                    day_data = cb_cache_df.loc[[d]]
                    if not self._is_cb_date_valid(d, day_data):
                        print(f"警告：交易日 {d} 的缓存数据不完整，将重新获取")
                        missing_dates.append(d)

        if missing_dates:
            print(f"发现 {len(missing_dates)} 个缺失或无效交易日，开始增量获取...")
            new_dfs = []
            for date in tqdm(missing_dates, desc="增量获取可转债数据"):
                df = self.get_cb_daily(date)
                if not df.empty:
                    df['trade_date'] = date
                    new_dfs.append(df)
                time.sleep(0.05)

            if new_dfs:
                new_df = pd.concat(new_dfs, ignore_index=True)
                if not cb_cache_df.empty and not force:
                    cb_cache_flat = cb_cache_df.reset_index()
                    combined = pd.concat([cb_cache_flat, new_df], ignore_index=True)
                else:
                    combined = new_df

                combined.drop_duplicates(subset=['ts_code', 'trade_date'], keep='last', inplace=True)
                combined['trade_date'] = pd.to_datetime(combined['trade_date'])
                combined.set_index('trade_date', inplace=True)
                combined.sort_index(inplace=True)
                self._save_cb_cache(combined)
                cb_cache_df = combined
            else:
                print("未获取到新数据")
        else:
            print("可转债数据已完整，无需更新")

        # 构建cb_data字典
        cb_data = {}
        if not cb_cache_df.empty:
            for date in trade_dates:
                if date in cb_cache_df.index:
                    day_data = cb_cache_df.loc[[date]].copy()
                    day_data = day_data.reset_index(drop=True)
                    cb_data[date] = day_data

        # ---------- 资产数据（全部Tushare）----------
        print("\n处理资产数据...")
        asset_data = {}
        for name, config in ASSET_CONFIG.items():
            if force:
                cache_series = pd.Series(dtype=float)
                missing_asset_dates = trade_dates
                print(f"强制模式：将重新获取 {name} 全部数据")
            else:
                cache_series = self._load_asset_cache(name)
                existing_dates_asset = set(cache_series.index) if not cache_series.empty else set()
                missing_asset_dates = [d for d in trade_dates if d not in existing_dates_asset]

            if missing_asset_dates:
                print(f"资产 {name} 缺失 {len(missing_asset_dates)} 个交易日，增量获取...")
                start_str = missing_asset_dates[0].strftime('%Y%m%d')
                end_str = missing_asset_dates[-1].strftime('%Y%m%d')

                new_series = self._get_tushare_asset_daily(name, config, start_str, end_str)

                if not new_series.empty:
                    # 仅保留缺失日期范围内的数据
                    new_series = new_series[new_series.index.isin(missing_asset_dates)]
                    combined = pd.concat([cache_series, new_series])
                    combined = combined[~combined.index.duplicated(keep='last')].sort_index()
                    self._save_asset_cache(name, combined)
                    asset_data[name] = combined
                else:
                    asset_data[name] = cache_series
            else:
                asset_data[name] = cache_series

            if asset_data[name].empty:
                print(f"⚠ 严重警告：{name} ({config['code']}) 无数据，请检查网络或数据源！")
            else:
                print(f"✓ {name} ({config['code']}) 加载成功，日期范围 {asset_data[name].index.min()} 至 {asset_data[name].index.max()}")

        # 对资产数据重新索引，确保每个交易日都有值（缺失填充NaN）
        for name in asset_data:
            asset_data[name] = asset_data[name].reindex(trade_dates, fill_value=np.nan)

        # ---------- 对资产价格进行清洗（插值修复） ----------
        print("\n清洗资产价格数据...")
        for name, series in asset_data.items():
            if series.empty:
                continue
            if name in ['Benchmark', 'Stock']:
                asset_data[name] = self._clean_price_series(
                    series, name,
                    max_change=0.2,
                    min_value=500,
                    max_value=20000
                )
            elif name == 'Cash':
                asset_data[name] = self._clean_price_series(
                    series, name,
                    max_change=0.2,
                    min_value=50,
                    max_value=200
                )
            else:
                asset_data[name] = self._clean_price_series(
                    series, name,
                    max_change=0.2,
                    min_value=0.1,
                    max_value=200
                )

        return {'cb_data': cb_data, 'asset_data': asset_data, 'trade_dates': trade_dates}


class DualLowStrategy:
    """双低因子可转债策略（日度版）"""

    def __init__(self, x_weight=X_WEIGHT, top_n=TOP_N):
        self.x_weight = x_weight
        self.top_n = top_n

    def calculate_double_low(self, price, premium):
        try:
            return float(price) + self.x_weight * float(premium) * 100
        except:
            return float('inf')

    def select_portfolio(self, df):
        """根据当日数据选出前N名转债，返回代码列表和双低值列表"""
        if df.empty or 'conv_premium' not in df.columns:
            return [], []
        df = df.copy()
        df['double_low'] = df.apply(lambda x: self.calculate_double_low(x['close'], x['conv_premium']), axis=1)
        df = df[df['double_low'] != float('inf')]
        df = df[df['double_low'] > 0]
        if df.empty:
            return [], []
        selected = df.nsmallest(self.top_n, 'double_low')
        return selected['ts_code'].tolist(), selected['double_low'].tolist()


class StopLossTakeProfitDaily:
    """
    止盈止损管理器（日度版）
    基于每日累计收益进行判断
    """
    def __init__(self, stop_loss, stop_gain, reentry_threshold):
        self.stop_loss = stop_loss
        self.stop_gain = stop_gain
        self.reentry_threshold = reentry_threshold

        self.state = 'closed'
        self.current_return = 0.0
        self.max_return = 0.0
        self.min_return_since_close = None
        self.cumulative_from_low = 0.0

    def update(self, daily_return, date):
        action = None
        if self.state == 'open':
            self.current_return += daily_return
            self.max_return = max(self.max_return, self.current_return)

            if self.current_return <= self.stop_loss:
                action = 'stop_loss'
                self.state = 'closed'
                self._reset_closed()
                print(f"  {date}: 触发止损 (累计收益 {self.current_return:.2%})")
            elif self.current_return <= self.max_return - self.stop_gain:
                action = 'take_profit'
                self.state = 'closed'
                self._reset_closed()
                print(f"  {date}: 触发止盈回撤 (累计收益 {self.current_return:.2%}, 峰值 {self.max_return:.2%})")
        else:
            if self.min_return_since_close is None or daily_return < self.min_return_since_close:
                self.min_return_since_close = daily_return
                self.cumulative_from_low = 0.0
            else:
                self.cumulative_from_low += daily_return

            if self.cumulative_from_low >= self.reentry_threshold:
                action = 'reentry'
                self.state = 'open'
                self._reset_open()
                print(f"  {date}: 重新开仓 (反弹累计 {self.cumulative_from_low:.2%})")
        return action

    def _reset_closed(self):
        self.min_return_since_close = None
        self.cumulative_from_low = 0.0

    def _reset_open(self):
        self.current_return = 0.0
        self.max_return = 0.0

    def get_effective_return(self, cb_return, harry_return):
        return cb_return if self.state == 'open' else harry_return


class DailyBacktestEngine:
    """日度回测引擎"""

    def __init__(self, data_loader, strategy, stop_loss, stop_gain, reentry_threshold):
        self.loader = data_loader
        self.strategy = strategy
        self.sltp = StopLossTakeProfitDaily(stop_loss, stop_gain, reentry_threshold)

        self.trade_dates = None
        self.cb_data = None
        self.asset_data = None

        self.results = []
        self.portfolio_history = {}

    def load_data(self, start_date, end_date, force=False):
        """加载数据并计算每周选债"""
        data = self.loader.get_daily_data(start_date, end_date, force=force)
        self.trade_dates = data['trade_dates']
        self.cb_data = data['cb_data']
        self.asset_data = data['asset_data']

        week_groups = {}
        for date in self.trade_dates:
            week_key = date.strftime('%Y-%W')
            if week_key not in week_groups:
                week_groups[week_key] = []
            week_groups[week_key].append(date)

        self.week_starts = {}
        self.week_ends = {}
        for week, dates in week_groups.items():
            start = dates[0]
            end = dates[-1]
            self.week_starts[week] = start
            self.week_ends[week] = end

        for week, start_date in self.week_starts.items():
            if start_date in self.cb_data:
                df = self.cb_data[start_date]
                bonds, _ = self.strategy.select_portfolio(df)
                self.portfolio_history[week] = bonds
            else:
                self.portfolio_history[week] = []

    def run(self):
        """执行回测，生成每日净值序列（包含哈利布朗组合净值）"""
        print("\n开始日度回测...")
        results = []

        # 准备资产收益率序列
        asset_returns = {}
        for asset, series in self.asset_data.items():
            if not series.empty:
                asset_returns[asset] = series.pct_change().dropna()
            else:
                asset_returns[asset] = pd.Series(dtype=float)

        benchmark_ret = asset_returns.get('Benchmark', pd.Series(dtype=float))

        # 基准净值（缺失时保持前值）
        benchmark_nav = pd.Series(index=self.trade_dates, dtype=float)
        benchmark_nav.iloc[0] = 1.0
        for i in range(1, len(self.trade_dates)):
            ret = benchmark_ret.get(self.trade_dates[i], np.nan)
            if pd.isna(ret):
                benchmark_nav.iloc[i] = benchmark_nav.iloc[i-1]
                print(f"警告：{self.trade_dates[i].date()} 沪深300收益率缺失，净值保持不变")
            else:
                benchmark_nav.iloc[i] = benchmark_nav.iloc[i-1] * (1 + ret)

        current_week = None
        current_bonds = []          # 本周选债列表（用于记录和开仓备选）
        prev_date = None
        portfolio_value = 1.0       # 策略净值
        portfolio_components = None  # 策略持仓 {'shares': dict, 'prices': dict}
        state = 'closed'

        # 纯双低组合独立变量
        cb_value = 1.0              # 纯双低净值
        cb_portfolio = None         # 纯双低持仓

        # 一直持有哈利布朗组合的净值
        harry_nav = 1.0

        for i, date in enumerate(self.trade_dates):
            week_key = date.strftime('%Y-%W')
            is_week_start = (week_key != current_week)   # 是否为每周第一个交易日

            # ---- 计算当日哈利布朗组合收益率（用于后续更新harry_nav） ----
            harry_ret = 0.0
            if i > 0 and prev_date is not None:
                ret_sum = 0.0
                total_w = 0.0
                for asset, weight in HARRY_WEIGHTS.items():
                    if asset in asset_returns and not asset_returns[asset].empty:
                        if date in asset_returns[asset].index:
                            ret = asset_returns[asset].loc[date]
                            ret_sum += weight * ret
                            total_w += weight
                if total_w > 0:
                    harry_ret = ret_sum / total_w
            # 哈利布朗净值更新
            harry_nav *= (1 + harry_ret)

            # ---- 先计算当日收益（用旧持仓） ----
            # 策略部分收益计算（基于 portfolio_components 的实际持仓）
            if state == 'open' and portfolio_components is not None:
                held_bonds = list(portfolio_components['shares'].keys())
                today_prices = {}
                for bond in held_bonds:
                    if date in self.cb_data and bond in self.cb_data[date].set_index('ts_code')['close']:
                        today_prices[bond] = self.cb_data[date].set_index('ts_code')['close'][bond]
                    else:
                        # 无数据时使用前一日价格（避免跳变）
                        today_prices[bond] = portfolio_components['prices'].get(bond, np.nan)

                # 计算价值（忽略价格无效的债券，但理论上不应发生）
                valid_prices = {b: p for b, p in today_prices.items() if not pd.isna(p)}
                if valid_prices:
                    value = sum(portfolio_components['shares'][b] * valid_prices[b] for b in valid_prices)
                    daily_return = value / portfolio_value - 1 if portfolio_value != 0 else 0.0
                    portfolio_value = value
                    # 更新 prices 字典
                    for b, p in valid_prices.items():
                        portfolio_components['prices'][b] = p
                else:
                    # 所有债券均无价格，净值不变（极罕见）
                    daily_return = 0.0
            else:
                daily_return = 0.0

            # 纯双低部分收益计算（基于 cb_portfolio 的实际持仓）
            if cb_portfolio is not None:
                held_cb_bonds = list(cb_portfolio['shares'].keys())
                today_prices_cb = {}
                for bond in held_cb_bonds:
                    if date in self.cb_data and bond in self.cb_data[date].set_index('ts_code')['close']:
                        today_prices_cb[bond] = self.cb_data[date].set_index('ts_code')['close'][bond]
                    else:
                        today_prices_cb[bond] = cb_portfolio['prices'].get(bond, np.nan)

                valid_prices_cb = {b: p for b, p in today_prices_cb.items() if not pd.isna(p)}
                if valid_prices_cb:
                    value_cb = sum(cb_portfolio['shares'][b] * valid_prices_cb[b] for b in valid_prices_cb)
                    cb_daily_return = value_cb / cb_value - 1 if cb_value != 0 else 0.0
                    cb_value = value_cb
                    for b, p in valid_prices_cb.items():
                        cb_portfolio['prices'][b] = p
                else:
                    cb_daily_return = 0.0
            else:
                cb_daily_return = 0.0

            # ---- 然后处理每周换仓（更新持仓） ----
            if is_week_start:
                current_week = week_key
                new_bonds = self.portfolio_history.get(week_key, [])
                # 更新纯双低持仓（始终基于新一周选债重建）
                if new_bonds:
                    # 获取有价格的有效债券
                    valid_new_bonds = []
                    prices = []
                    for bond in new_bonds:
                        if date in self.cb_data and bond in self.cb_data[date].set_index('ts_code')['close']:
                            p = self.cb_data[date].set_index('ts_code')['close'][bond]
                            valid_new_bonds.append(bond)
                            prices.append(p)
                    if valid_new_bonds:
                        n = len(valid_new_bonds)
                        shares = {bond: (cb_value / n) / p for bond, p in zip(valid_new_bonds, prices)}
                        cb_portfolio = {'shares': shares, 'prices': dict(zip(valid_new_bonds, prices))}
                    else:
                        cb_portfolio = None   # 无可买债券，现金
                else:
                    cb_portfolio = None

                # 更新策略选债列表（current_bonds 现在记录本周选债）
                current_bonds = new_bonds
                # 如果策略处于开仓状态，需要重新建仓（清空旧持仓）
                if state == 'open':
                    portfolio_components = None   # 强制当天后续重新建仓

            # ---- 策略部分：处理开仓/闭仓及止盈止损 ----
            if state == 'open':
                # 如果需要建仓（刚开仓或刚换仓）
                if portfolio_components is None:
                    if current_bonds:
                        # 用当日价格建仓，只保留有价格的债券
                        valid_bonds = []
                        prices = []
                        for bond in current_bonds:
                            if date in self.cb_data and bond in self.cb_data[date].set_index('ts_code')['close']:
                                p = self.cb_data[date].set_index('ts_code')['close'][bond]
                                valid_bonds.append(bond)
                                prices.append(p)
                        if prices:
                            n = len(prices)
                            shares = {bond: (portfolio_value / n) / p for bond, p in zip(valid_bonds, prices)}
                            portfolio_components = {'shares': shares, 'prices': dict(zip(valid_bonds, prices))}
                            # 更新 current_bonds 为实际持有的债券（便于记录和后续换仓）
                            current_bonds = valid_bonds
                            daily_return = 0.0   # 建仓日收益已在前面计算过旧持仓，此处设为0避免重复
                        else:
                            # 无任何有效转债，强制闭仓
                            state = 'closed'
                            portfolio_components = None
                            current_bonds = []   # 清空记录
                    else:
                        # 本周无选债，无法开仓
                        state = 'closed'
                        portfolio_components = None

                # 如果已有持仓（已在上方计算过收益），则检查止盈止损
                if portfolio_components is not None:
                    effective_return = daily_return
                    action = self.sltp.update(daily_return, date)
                    if action in ['stop_loss', 'take_profit']:
                        state = 'closed'
                        portfolio_components = None
                else:
                    effective_return = 0.0   # 无持仓时收益为0
            else:  # closed
                # 策略处于闭仓时，实际收益就是哈利布朗收益
                effective_return = harry_ret
                portfolio_value *= (1 + harry_ret)   # 净值按哈利布朗组合更新
                daily_return = 0.0                     # 转债日收益为0

                # 监测反弹，决定是否重新开仓
                action = self.sltp.update(harry_ret, date)
                if action == 'reentry':
                    state = 'open'
                    portfolio_components = None   # 准备开仓建仓（将在下一交易日进行）

            results.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'cb_value': cb_value,
                'harry_nav': harry_nav,
                'daily_return': daily_return,
                'effective_return': effective_return,
                'state': state,
                'action': action if action else None,
                'bonds': current_bonds.copy() if current_bonds else [],
                'benchmark_return': benchmark_ret[date] if date in benchmark_ret.index else np.nan,
                'benchmark_nav': benchmark_nav[date]
            })
            prev_date = date

        self.results = pd.DataFrame(results)
        self.results.set_index('date', inplace=True)
        self.results.sort_index(inplace=True)
        self.results['cumulative_strategy'] = self.results['portfolio_value']
        self.results['cumulative_benchmark'] = self.results['benchmark_nav']
        self.results['cumulative_cb'] = self.results['cb_value']
        self.results['cumulative_harry'] = self.results['harry_nav']
        return self.results

    def get_next_day_advice(self):
        last_date = self.results.index[-1]
        week_key = last_date.strftime('%Y-%W')
        current_bonds = self.portfolio_history.get(week_key, [])
        state = self.results.loc[last_date, 'state']
        if state == 'open':
            bonds = current_bonds
            reason = "策略处于开仓状态，继续持有本周选债"
        else:
            bonds = []
            reason = "策略处于平仓状态，建议持有哈利布朗组合"
        return bonds, reason

    def get_latest_weights(self):
        last_date = self.results.index[-1]
        week_key = last_date.strftime('%Y-%W')
        bonds = self.portfolio_history.get(week_key, [])
        if not bonds:
            return {}
        return {bond: 1.0/len(bonds) for bond in bonds}


def calculate_performance_metrics(returns, benchmark_returns=None, days_per_year=252):
    if len(returns) == 0:
        return {}
    nav = (1 + returns).cumprod()
    total_ret = nav.iloc[-1] - 1
    years = len(returns) / days_per_year
    ann_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0
    ann_vol = returns.std() * np.sqrt(days_per_year)
    sharpe = ann_ret / ann_vol if ann_vol != 0 else 0

    rolling_max = nav.cummax()
    drawdown = (nav - rolling_max) / rolling_max
    max_dd = drawdown.min()
    if max_dd < 0:
        start = drawdown[drawdown == max_dd].index[0]
        prev_peak = rolling_max[:start].idxmax()
        end = start
        after = drawdown[drawdown.index > start]
        recovery = after[after == 0]
        if len(recovery) > 0:
            end = recovery.index[0]
        max_dd_period = f"{prev_peak.date()} 至 {end.date()} ({(end - prev_peak).days} 天)"
    else:
        max_dd_period = "无"

    win_rate = (returns > 0).mean()
    avg_win = returns[returns > 0].mean()
    avg_loss = -returns[returns < 0].mean()
    pl_ratio = avg_win / avg_loss if avg_loss != 0 else 0

    metrics = {
        '年化收益率': f"{ann_ret:.2%}",
        '年化波动率': f"{ann_vol:.2%}",
        '夏普比率': f"{sharpe:.2f}",
        '最大回撤': f"{max_dd:.2%}",
        '最大回撤区间': max_dd_period,
        '胜率': f"{win_rate:.2%}",
        '盈亏比': f"{pl_ratio:.2f}"
    }
    return metrics


def plot_nav(results, title="策略净值曲线"):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    ax = axes[0]
    ax.plot(results.index, results['cumulative_strategy'], label='策略净值', linewidth=2, color='blue')
    benchmark_valid = results['cumulative_benchmark'].notna().any() and not (results['cumulative_benchmark'] == 1.0).all()
    if benchmark_valid:
        ax.plot(results.index, results['cumulative_benchmark'], label='沪深300指数', linewidth=2, color='red', alpha=0.7)
    else:
        ax.text(0.5, 0.5, '沪深300数据缺失', transform=ax.transAxes, ha='center', va='center', fontsize=12, color='red', alpha=0.5)

    ax.plot(results.index, results['cumulative_cb'], label='纯双低组合', linewidth=2, color='green', alpha=0.5)
    ax.plot(results.index, results['cumulative_harry'], label='哈利布朗组合', linewidth=2, color='purple', alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel('日期')
    ax.set_ylabel('净值')
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.hist(results['effective_return'] * 100, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(results['effective_return'].mean() * 100, color='red', linestyle='--', label=f"均值 {results['effective_return'].mean()*100:.2f}%")
    ax.set_title('策略日收益率分布')
    ax.set_xlabel('收益率 (%)')
    ax.set_ylabel('频率')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    return fig


def generate_html_report(engine, results, metrics, last_date):
    weights = engine.get_latest_weights()
    bonds, reason = engine.get_next_day_advice()

    # 获取上一日持仓
    if len(results) > 1:
        prev_date = results.index[-2]
        prev_bonds = results.loc[prev_date, 'bonds']
        prev_bonds_str = ', '.join(prev_bonds) if prev_bonds else '无持仓'
    else:
        prev_date = '无'
        prev_bonds_str = '无历史数据'

    benchmark_nav = results['cumulative_benchmark'].iloc[-1]
    if pd.isna(benchmark_nav) or benchmark_nav == 1.0:
        benchmark_display = "数据缺失"
    else:
        benchmark_display = f"{benchmark_nav:.4f}"

    html = f"""
    <html>
    <head>
        <meta charset="UTF-8">
        <title>双低可转债策略日报 - {last_date}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
            h2 {{ color: #4CAF50; margin-top: 30px; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 15px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .metric {{ font-weight: bold; color: #333; }}
            .footer {{ margin-top: 40px; font-size: 12px; color: #666; }}
        </style>
    </head>
    <body>
        <h1>双低可转债策略日报 - {last_date}</h1>

        <h2>最新持仓 (本周选债)</h2>
        <table>
            <tr><th>转债代码</th><th>权重</th></tr>
    """
    for bond, w in weights.items():
        html += f"<tr><td>{bond}</td><td>{w:.2%}</td></tr>"
    if not weights:
        html += "<tr><td colspan='2'>当前无持仓（平仓状态）</td></tr>"

    html += f"""
        </table>

        <h2>上一日持仓 ({prev_date})</h2>
        <p>{prev_bonds_str}</p>

        <h2>下一交易日建议</h2>
        <p><strong>建议持仓:</strong> {', '.join(bonds) if bonds else '持有现金（哈利布朗组合）'}</p>
        <p><strong>理由:</strong> {reason}</p>

        <h2>最新净值</h2>
        <p>策略净值: {results['cumulative_strategy'].iloc[-1]:.4f}</p>
        <p>沪深300净值: {benchmark_display}</p>
        <p>纯双低组合净值: {results['cumulative_cb'].iloc[-1]:.4f}</p>
        <p>哈利布朗组合净值: {results['cumulative_harry'].iloc[-1]:.4f}</p>

        <h2>绩效指标</h2>
        <table>
            <tr><th>指标</th><th>值</th></tr>
    """
    for k, v in metrics.items():
        html += f"<tr><td>{k}</td><td>{v}</td></tr>"

    fig = plot_nav(results)
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    html += f"""
        </table>
        <img src="data:image/png;base64,{img_base64}" style="width:100%; max-width:1200px;">

        <div class="footer">
            <p>报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p>数据范围: {results.index[0].date()} 至 {results.index[-1].date()}</p>
        </div>
    </body>
    </html>
    """
    return html

def main():
    parser = argparse.ArgumentParser(description='双低可转债策略每日运行')
    parser.add_argument('--start', type=str, default=START_DATE, help='回测起始日期，格式YYYYMMDD')
    parser.add_argument('--end', type=str, default=END_DATE, help='回测结束日期，格式YYYYMMDD')
    parser.add_argument('--no_email', action='store_true', help='不发送邮件')
    parser.add_argument('--force', action='store_true', help='强制重新获取所有数据（忽略缓存）')
    args = parser.parse_args()

    print("=" * 70)
    print("双低可转债策略每日运行（Tushare全数据 + 价格插值修复 + 现金资产完善）")
    print("=" * 70)

    # 初始化
    loader = DataLoader(TUSHARE_TOKEN)
    strategy = DualLowStrategy(X_WEIGHT, TOP_N)

    # 加载数据并运行回测
    engine = DailyBacktestEngine(loader, strategy, STOP_LOSS, STOP_GAIN, REENTRY_THRESHOLD)
    engine.load_data(args.start, args.end, force=args.force)
    results = engine.run()

    if results.empty:
        print("回测结果为空，退出。")
        return

    metrics = calculate_performance_metrics(results['effective_return'], results['benchmark_return'])

    last_date = results.index[-1].date()
    print(f"\n最新日期: {last_date}")
    print(f"策略净值: {results['cumulative_strategy'].iloc[-1]:.4f}")
    print(f"沪深300净值: {results['cumulative_benchmark'].iloc[-1]:.4f}")
    print(f"纯双低净值: {results['cumulative_cb'].iloc[-1]:.4f}")
    print(f"哈利布朗净值: {results['cumulative_harry'].iloc[-1]:.4f}")

    strategy_name = "双低可转债策略"
    current_date = datetime.now().strftime('%Y%m%d')
    report_filename = f"{strategy_name}.html"
    html_content = generate_html_report(engine, results, metrics, last_date)
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"\nHTML报告已保存至 {report_filename}")

    # nav_df = results[['cumulative_strategy', 'cumulative_benchmark', 'cumulative_cb', 'cumulative_harry',
    #                   'effective_return', 'benchmark_return', 'daily_return']].copy()
    # nav_df.columns = ['策略净值', '沪深300净值', '纯双低净值', '哈利布朗净值',
    #                   '策略日收益率', '沪深300日收益率', '纯双低日收益率']
    # nav_df.index = nav_df.index.date
    # nav_df.index.name = '日期'
    # nav_excel = f"{strategy_name}_净值曲线.xlsx"
    # nav_df.to_excel(nav_excel, engine='openpyxl')
    # print(f"净值曲线数据已保存至 {nav_excel}")

    # holdings_df = results[['bonds', 'state']].copy()
    # holdings_df['持仓代码'] = holdings_df['bonds'].apply(lambda x: ', '.join(x) if x else '无持仓')
    # holdings_df['持仓数量'] = holdings_df['bonds'].apply(len)
    # holdings_df['状态'] = holdings_df['state']
    # holdings_df = holdings_df[['持仓代码', '持仓数量', '状态']]
    # holdings_df.index = holdings_df.index.date
    # holdings_df.index.name = '日期'
    # holdings_excel = f"{strategy_name}_每日持仓.xlsx"
    # holdings_df.to_excel(holdings_excel, engine='openpyxl')
    # print(f"每日持仓数据已保存至 {holdings_excel}")


    # 发送邮件
    print("\n" + "=" * 70)
    print("发送邮件...")
    print("=" * 70)

    # send emails
    HTML_PATH = "双低可转债策略.html"
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
            subject="双低可转债策略 Daily Backtest Report",
            html_body=HTML_BODY,
            attachment_path=HTML_PATH
        )

    print("\n" + "=" * 70)
    print("任务完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()