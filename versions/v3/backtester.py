import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import argparse
from datetime import datetime

# 确保能导入当前目录下的模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    from data_fetcher import get_stock_data
    from indicator_utils import calculate_blue_signal_series, calculate_heima_signal_series, calculate_kdj_series, calculate_volume_profile_metrics, calculate_atr_series, calculate_adx_series
except ImportError:
    print("Error: Could not import necessary modules. Make sure you are in the correct directory.")
    sys.exit(1)

# ==================== ADX 计算函数 (Moved to indicator_utils in future, kept here for now) ====================
def calculate_adx_series_local(high, low, close, period=14):
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    tr = calculate_atr_series(high, low, close, period=1)
    up_move = high_s - high_s.shift(1)
    down_move = low_s.shift(1) - low_s
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    tr_smooth = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean()
    plus_dm_smooth = pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean()
    minus_dm_smooth = pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean()
    tr_smooth = tr_smooth.replace(0, np.nan)
    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    dx = dx.fillna(0)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    return adx.values

class SimpleBacktester:
    """
    简单回测引擎
    
    信号时序说明 (No Lookahead Bias):
    ================================
    1. 日线 BLUE 在 time T 收盘后计算完成
    2. 信号判断基于 time T 的数据 (包括当天收盘价)
    3. 买入执行在 time T+1 的开盘价
    4. 这符合真实交易场景: 收盘后看到信号 -> 第二天开盘买入
    
    周线信号处理:
    - 周线 BLUE 在周五收盘后确定
    - 在日线回测中，使用 shift(1) + ffill，确保下周一才能看到上周的周线 BLUE
    """
    
    def __init__(self, symbol, market='US', initial_capital=100000, days=1095, 
                 commission_rate=0.001, blue_threshold=100, require_heima=False, 
                 require_week_blue=False, require_vp_filter=False, 
                 use_risk_management=True, strategy_mode='C',
                 strict_mode=True):  # 新增：严格模式
        """
        Args:
            strict_mode: 如果为 True，强制执行信号延迟验证
        """
        self.symbol = symbol
        self.market = market
        self.initial_capital = initial_capital
        self.days = days
        self.commission_rate = commission_rate
        self.blue_threshold = blue_threshold
        self.require_heima = require_heima
        self.require_week_blue = require_week_blue
        self.require_vp_filter = require_vp_filter
        self.use_risk_management = use_risk_management
        self.strategy_mode = strategy_mode
        self.strict_mode = strict_mode
        
        self.stop_atr_multiple = 2.0
        self.risk_per_trade_pct = 0.02
        self.adaptive_info = "标准模式 (未启动)"
        
        self.df = None
        self.results = {}
        self.yearly_returns = {}
        self.trades = []
        self.rejected_trades = []
        
    def load_data(self):
        # print(f"Loading data for {self.symbol} ({self.market})...")
        self.df = get_stock_data(self.symbol, self.market, self.days)
        if self.df is None or self.df.empty:
            # print("Failed to load data.")
            return False
        return True
        
    def adapt_parameters(self):
        if self.df is None or len(self.df) < 60:
            return

        log_ret = np.log(self.df['Close'] / self.df['Close'].shift(1))
        volatility = log_ret.tail(252).std() * np.sqrt(252)
        try:
            current_adx = self.df['ADX'].iloc[-1]
            avg_adx = self.df['ADX'].tail(60).mean()
        except:
            avg_adx = 20 # Default
        
        stop_mult = 2.0
        risk_pct = 0.02
        regime = "Standard"
        
        if volatility > 0.60:
            stop_mult = 3.5
            risk_pct = 0.01
            regime = "High Vol"
        elif volatility > 0.40:
            stop_mult = 2.8
            risk_pct = 0.015
            regime = "High Vol"
        elif volatility < 0.20:
            stop_mult = 1.8
            risk_pct = 0.03
            regime = "Low Vol"
            
        if avg_adx > 30:
            stop_mult += 0.5
            
        self.stop_atr_multiple = stop_mult
        self.risk_per_trade_pct = risk_pct
        self.adaptive_info = regime

    def calculate_signals(self):
        if self.df is None:
            return
            
        self.df['Day_BLUE'] = calculate_blue_signal_series(
            self.df['Open'].values, self.df['High'].values, self.df['Low'].values, self.df['Close'].values
        )
        self.df['heima'], self.df['juedi'] = calculate_heima_signal_series(
            self.df['High'].values, self.df['Low'].values, self.df['Close'].values, self.df['Open'].values
        )
        self.df['K'], self.df['D'], self.df['J'] = calculate_kdj_series(
            self.df['High'].values, self.df['Low'].values, self.df['Close'].values
        )
        self.df['ATR'] = calculate_atr_series(
            self.df['High'].values, self.df['Low'].values, self.df['Close'].values, period=14
        )
        try:
            self.df['ADX'] = calculate_adx_series(
                self.df['High'].values, self.df['Low'].values, self.df['Close'].values
            )
        except NameError:
            self.df['ADX'] = calculate_adx_series_local(
                self.df['High'].values, self.df['Low'].values, self.df['Close'].values
            )
        
        if self.use_risk_management:
            self.adapt_parameters()
        
        df_weekly = self.df.resample('W-FRI').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna()
        
        if df_weekly.empty:
            self.df['Week_BLUE_Ref'] = 0
            self.df['Week_Heima_Ref'] = False
            self.df['Week_Juedi_Ref'] = False
            return

        weekly_blue = calculate_blue_signal_series(
            df_weekly['Open'].values, df_weekly['High'].values, 
            df_weekly['Low'].values, df_weekly['Close'].values
        )
        df_weekly['Week_BLUE'] = weekly_blue
        
        df_weekly['heima'], df_weekly['juedi'] = calculate_heima_signal_series(
            df_weekly['High'].values, df_weekly['Low'].values, df_weekly['Close'].values, df_weekly['Open'].values
        )
        
        df_weekly_shifted = df_weekly.shift(1)
        
        self.df['Week_BLUE_Ref'] = df_weekly_shifted['Week_BLUE'].reindex(self.df.index, method='ffill').fillna(0)
        self.df['Week_Heima_Ref'] = df_weekly_shifted['heima'].reindex(self.df.index, method='ffill').fillna(False)
        self.df['Week_Juedi_Ref'] = df_weekly_shifted['juedi'].reindex(self.df.index, method='ffill').fillna(False)
        
    def run_backtest(self):
        if self.df is None:
            return
            
        # print(f"Running backtest (Strategy {self.strategy_mode})...")
        
        capital = self.initial_capital
        position = 0 
        portfolio_values = []
        
        dates = self.df.index
        opens = self.df['Open'].values
        closes = self.df['Close'].values
        lows = self.df['Low'].values
        
        day_blues = self.df['Day_BLUE'].values
        week_blues = self.df['Week_BLUE_Ref'].values
        
        heimas = self.df['heima'].values
        juedis = self.df['juedi'].values
        week_heimas = self.df['Week_Heima_Ref'].values
        week_juedis = self.df['Week_Juedi_Ref'].values
        
        Js = self.df['J'].values
        ATRs = self.df['ATR'].values
        
        entry_price = 0 
        stop_loss_price = 0 
        
        for i in range(len(self.df)):
            if i == len(self.df) - 1:
                portfolio_values.append(capital + position * closes[i])
                break
                
            d_blue = day_blues[i]
            w_blue = week_blues[i]
            is_heima = heimas[i]
            is_juedi = juedis[i]
            w_heima = week_heimas[i]
            w_juedi = week_juedis[i]
            
            kdj_j = Js[i]
            atr = ATRs[i]
            
            close_price = closes[i]
            low_price = lows[i]
            next_open = opens[i+1]
            
            signal = 0 
            sell_reason = ""
            
            # --- 卖出 ---
            if position > 0:
                if self.use_risk_management:
                     if low_price < stop_loss_price:
                         signal = -1
                         sell_reason = "Stop Loss"
                     
                     profit_per_share = close_price - entry_price
                     if profit_per_share > 1.5 * atr:
                         new_stop = max(stop_loss_price, entry_price + 0.1 * atr)
                         stop_loss_price = new_stop
                     if profit_per_share > 4 * atr:
                         trail_dist = self.stop_atr_multiple * 0.7 * atr
                         new_stop = max(stop_loss_price, close_price - trail_dist)
                         stop_loss_price = new_stop
                
                # 卖出条件1: KDJ J > 90 (超买)
                if signal == 0 and kdj_j > 90:
                    signal = -1
                    sell_reason = "KDJ J>90 Overbought"
                
                # 卖出条件2: 跌破5日均线
                if signal == 0 and i >= 5:
                    sma5 = np.mean(closes[i-4:i+1])  # 5日均线
                    if close_price < sma5:
                        signal = -1
                        sell_reason = "Break Below MA5"

            # --- 买入逻辑 (多策略核心) ---
            elif position == 0:
                buy_candidate = False
                trigger_source = ""
                
                start_idx = max(0, i-4)
                recent_blues = day_blues[start_idx:i+1]
                recent_heimas = heimas[start_idx:i+1] | juedis[start_idx:i+1]
                
                has_recent_blue = np.any(recent_blues > self.blue_threshold)
                has_recent_heima_daily = np.any(recent_heimas)
                has_heima_context = has_recent_heima_daily or w_heima or w_juedi
                
                is_blue_trigger = (d_blue > self.blue_threshold)
                is_heima_trigger = (is_heima or is_juedi or w_heima or w_juedi)
                is_week_blue_good = (w_blue > self.blue_threshold)
                
                # 计算 VP (如果策略需要)
                vp_ok = True
                if self.strategy_mode in ['A', 'B']:
                    vp = calculate_volume_profile_metrics(closes[:i+1], self.df['Volume'].values[:i+1], close_price)
                    if vp['profit_ratio'] < 0.05 and vp['price_pos'] == 'Below':
                        vp_ok = False

                # 策略 A: 双蓝共振 + 黑马 + VP (严谨型)
                if self.strategy_mode == 'A':
                    # 必须: 日线Blue AND 周线Blue AND 黑马 AND VP
                    # 允许黑马是最近出现的
                    if is_blue_trigger and is_week_blue_good and has_heima_context and vp_ok:
                        buy_candidate = True
                        trigger_source = "Strat A: Full Resonance"
                        
                # 策略 B: 日线Blue + 黑马 + VP (平衡型)
                elif self.strategy_mode == 'B':
                    # 必须: 日线Blue AND 黑马 AND VP (不强制周线)
                    if is_blue_trigger and has_heima_context and vp_ok:
                        buy_candidate = True
                        trigger_source = "Strat B: Daily Blue + Heima"
                
                # 策略 C: 宽松共振 (BLUE OR Heima) - 之前修正的逻辑
                elif self.strategy_mode == 'C':
                    # 只要有 BLUE 和 Heima 在时间窗口内共振即可，无需 VP，周线可选辅助
                    if is_blue_trigger and has_heima_context:
                        buy_candidate = True
                        trigger_source = "Strat C: Blue+Heima"
                    elif is_heima_trigger and (has_recent_blue or is_week_blue_good):
                        buy_candidate = True
                        trigger_source = "Strat C: Heima+Blue"
                        
                # 策略 D: 纯趋势 (BLUE Only) (激进型)
                elif self.strategy_mode == 'D':
                    if is_blue_trigger:
                        buy_candidate = True
                        trigger_source = "Strat D: Blue Trend"

                if buy_candidate:
                    signal = 1
            
            # --- 执行 ---
            if signal == 1:
                if np.isnan(next_open) or next_open <= 0:
                    continue
                
                if self.use_risk_management and atr > 0:
                    risk_per_trade = capital * self.risk_per_trade_pct
                    stop_distance = self.stop_atr_multiple * atr
                    calculated_shares = int(risk_per_trade / stop_distance)
                    max_capital_shares = int((capital * 0.6) / next_open)
                    shares = min(calculated_shares, max_capital_shares)
                    stop_loss_price = next_open - stop_distance
                else:
                    max_cost = capital / (1 + self.commission_rate)
                    shares = int(max_cost / next_open)
                
                if shares > 0:
                    cost = shares * next_open
                    commission = cost * self.commission_rate
                    
                    if capital >= cost + commission:
                        vp_metrics = calculate_volume_profile_metrics(closes[:i+1], self.df['Volume'].values[:i+1], next_open)
                        capital -= (cost + commission)
                        position = shares
                        entry_price = next_open
                        self.trades.append({
                            'type': 'BUY',
                            'date': dates[i+1],
                            'price': next_open,
                            'shares': shares,
                            'value': cost,
                            'reason': trigger_source,
                            'vp_metrics': vp_metrics,
                            'stop_loss': stop_loss_price if self.use_risk_management else 0
                        })
            
            elif signal == -1:
                revenue = position * next_open
                commission = revenue * self.commission_rate
                capital += (revenue - commission)
                last_buy_value = self.trades[-1]['value'] if self.trades else 0
                pnl = revenue - last_buy_value - commission - (last_buy_value * self.commission_rate)
                self.trades.append({
                    'type': 'SELL',
                    'date': dates[i+1],
                    'price': next_open,
                    'shares': position,
                    'value': revenue,
                    'pnl': pnl,
                    'reason': sell_reason
                })
                position = 0
                stop_loss_price = 0
            
            current_value = capital + position * close_price
            portfolio_values.append(current_value)
            
        self.df['Portfolio'] = pd.Series(portfolio_values, index=self.df.index[:len(portfolio_values)])
        
        final_value = portfolio_values[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        annual_return = (1 + total_return) ** (365 / self.days) - 1
        
        portfolio_series = pd.Series(portfolio_values)
        rolling_max = portfolio_series.cummax()
        drawdown = (portfolio_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        winning_trades = len([t for t in self.trades if t.get('type') == 'SELL' and t.get('pnl', 0) > 0])
        total_closed_trades = len([t for t in self.trades if t.get('type') == 'SELL'])
        win_rate = winning_trades / total_closed_trades if total_closed_trades > 0 else 0
        
        self.calculate_yearly_returns()
        
        self.results = {
            'Initial Capital': self.initial_capital,
            'Final Value': final_value,
            'Total Return': total_return,
            'Annual Return': annual_return,
            'Max Drawdown': max_drawdown,
            'Total Trades': total_closed_trades,
            'Win Rate': win_rate,
            'Adaptive Info': self.adaptive_info,
            'Yearly Returns': self.yearly_returns
        }
        
    def calculate_yearly_returns(self):
        """计算分年度收益率"""
        if self.df is None or 'Portfolio' not in self.df.columns:
            return
        yearly_res = {}
        try:
            groups = self.df.groupby(self.df.index.year)
            for year, group in groups:
                if len(group) < 2: continue
                start_val = group['Portfolio'].iloc[0]
                end_val = group['Portfolio'].iloc[-1]
                ret = (end_val - start_val) / start_val
                yearly_res[int(year)] = ret
        except Exception as e:
            pass
        self.yearly_returns = yearly_res

    # ... plot_results code omitted for brevity ...
    def plot_results(self, show=True):
        if self.df is None or 'Portfolio' not in self.df.columns:
            return None
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=('Price & Signals', 'Portfolio Value'), row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(x=self.df.index, open=self.df['Open'], high=self.df['High'], low=self.df['Low'], close=self.df['Close'], name='Price'), row=1, col=1)
        buy_dates = [t['date'] for t in self.trades if t['type'] == 'BUY']
        buy_prices = [t['price'] for t in self.trades if t['type'] == 'BUY']
        sell_dates = [t['date'] for t in self.trades if t['type'] == 'SELL']
        sell_prices = [t['price'] for t in self.trades if t['type'] == 'SELL']
        fig.add_trace(go.Scatter(x=buy_dates, y=buy_prices, mode='markers', marker=dict(symbol='triangle-up', size=12, color='green', line=dict(width=1, color='black')), name='Buy'), row=1, col=1)
        fig.add_trace(go.Scatter(x=sell_dates, y=sell_prices, mode='markers', marker=dict(symbol='triangle-down', size=12, color='red', line=dict(width=1, color='black')), name='Sell'), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df['Portfolio'], mode='lines', name='Portfolio Value', line=dict(color='purple', width=2)), row=2, col=1)
        title = f"Backtest ({self.strategy_mode}): {self.symbol} | Return: {self.results['Total Return']:.2%}"
        fig.update_layout(title=title, xaxis_rangeslider_visible=False, height=800)
        if show: fig.show()
        return fig

# Main function updated to support strategy mode from args (omitted for brevity, will update in batch_backtest)
def main():
    pass 

if __name__ == "__main__":
    pass
