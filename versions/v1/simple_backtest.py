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
    from indicator_utils import calculate_blue_signal_series, calculate_heima_signal_series, calculate_kdj_series, calculate_volume_profile_metrics
except ImportError:
    print("Error: Could not import necessary modules. Make sure you are in the correct directory.")
    sys.exit(1)

class SimpleBacktester:
    def __init__(self, symbol, market='US', initial_capital=100000, days=1095, commission_rate=0.001, blue_threshold=100, require_heima=False, require_week_blue=False, require_vp_filter=False):
        self.symbol = symbol
        self.market = market
        self.initial_capital = initial_capital
        self.days = days
        self.commission_rate = commission_rate
        self.blue_threshold = blue_threshold
        self.require_heima = require_heima
        self.require_week_blue = require_week_blue
        self.require_vp_filter = require_vp_filter
        self.df = None
        self.results = {}
        self.trades = []
        self.rejected_trades = [] # 记录被过滤的交易信号
        
    def load_data(self):
        print(f"Loading data for {self.symbol} ({self.market})...")
        self.df = get_stock_data(self.symbol, self.market, self.days)
        if self.df is None or self.df.empty:
            print("Failed to load data.")
            return False
        return True
        
    def calculate_signals(self):
        if self.df is None:
            return
            
        # 1. 计算日线指标
        self.df['Day_BLUE'] = calculate_blue_signal_series(
            self.df['Open'].values, self.df['High'].values, self.df['Low'].values, self.df['Close'].values
        )
        self.df['heima'], self.df['juedi'] = calculate_heima_signal_series(
            self.df['High'].values, self.df['Low'].values, self.df['Close'].values, self.df['Open'].values
        )
        # 计算 KDJ (用于卖出)
        self.df['K'], self.df['D'], self.df['J'] = calculate_kdj_series(
            self.df['High'].values, self.df['Low'].values, self.df['Close'].values
        )
        
        # 2. 计算周线指标
        # 重新采样为周线 (W-FRI: 每周五作为结束)
        df_weekly = self.df.resample('W-FRI').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna()
        
        if df_weekly.empty:
            self.df['Week_BLUE_Ref'] = 0
            return

        weekly_blue = calculate_blue_signal_series(
            df_weekly['Open'].values, df_weekly['High'].values, 
            df_weekly['Low'].values, df_weekly['Close'].values
        )
        df_weekly['Week_BLUE'] = weekly_blue
        
        # 3. 将周线指标映射回日线
        # 使用 shift(1) 避免未来函数：本周每天只能看到"上周"确定的周线值
        # 这样更加严谨
        df_weekly_shifted = df_weekly.shift(1)
        
        # reindex 并 ffill: 将上周五的值填充到本周的每一天
        self.df['Week_BLUE_Ref'] = df_weekly_shifted['Week_BLUE'].reindex(self.df.index, method='ffill')
        self.df['Week_BLUE_Ref'] = self.df['Week_BLUE_Ref'].fillna(0)
        
        # 调试输出
        print("\n[DEBUG] Data Sample (Last 10 days):")
        print(self.df[['Close', 'Day_BLUE', 'Week_BLUE_Ref']].tail(10))
        print(f"\n[DEBUG] Max Day BLUE: {self.df['Day_BLUE'].max()}")
        print(f"[DEBUG] Max Week BLUE Ref: {self.df['Week_BLUE_Ref'].max()}")
        
    def run_backtest(self):
        if self.df is None:
            return
            
        print("Running backtest (Strategy: Dual BLUE + Heima Entry, KDJ Exit)...")
        
        capital = self.initial_capital
        position = 0
        portfolio_values = []
        
        # 记录每一步
        dates = self.df.index
        opens = self.df['Open'].values
        closes = self.df['Close'].values
        
        day_blues = self.df['Day_BLUE'].values
        week_blues = self.df['Week_BLUE_Ref'].values
        heimas = self.df['heima'].values
        juedis = self.df['juedi'].values
        Js = self.df['J'].values
        
        entry_price = 0 # 记录持仓成本
        
        for i in range(len(self.df)):
            if i == len(self.df) - 1:
                portfolio_values.append(capital + position * closes[i])
                break
                
            # 获取当日信号
            d_blue = day_blues[i]
            w_blue = week_blues[i]
            is_heima = heimas[i]
            is_juedi = juedis[i]
            kdj_j = Js[i]
            
            close_price = closes[i]
            next_open = opens[i+1] # 假设次日开盘交易
            
            signal = 0 
            
            # --- 卖出逻辑 ---
            # 持仓时，如果 J > 100，卖出
            # (也可以增加止损逻辑，这里先只实现 KDJ 卖出)
            if position > 0:
                if kdj_j > 100:
                    signal = -1
            
            # --- 买入逻辑 ---
            # 空仓时
            elif position == 0:
                # 基础条件: 日线 BLUE > 阈值
                if d_blue > self.blue_threshold:
                    buy_condition = True
                    reject_reason = None
                    
                    # 可选条件: 周线 BLUE
                    if self.require_week_blue and not (w_blue > self.blue_threshold):
                        buy_condition = False
                        reject_reason = f"周线BLUE不足 ({w_blue:.1f})"
                        
                    # 可选条件: 黑马/掘底 (如果没有被周线拒绝)
                    elif self.require_heima and not (is_heima or is_juedi):
                        buy_condition = False
                        reject_reason = "无黑马/掘底信号"
                        
                    # 可选条件: 筹码分布过滤 (如果没有被前面拒绝)
                    elif self.require_vp_filter:
                        # 计算当前 VP
                        vp = calculate_volume_profile_metrics(
                            closes[:i+1], self.df['Volume'].values[:i+1], close_price
                        )
                        # 过滤逻辑
                        if vp['profit_ratio'] < 0.05 and vp['price_pos'] == 'Below':
                            buy_condition = False
                            reject_reason = f"筹码差: 获利盘{vp['profit_ratio']:.1%}且受压"
                
                    if buy_condition:
                        signal = 1
                    elif reject_reason:
                        # 记录被拒绝的信号
                        self.rejected_trades.append({
                            'date': dates[i],
                            'price': close_price,
                            'reason': reject_reason,
                            'blue': d_blue
                        })
            
            # --- 执行交易 ---
            if signal == 1:
                # 全仓买入 (考虑佣金)
                if np.isnan(next_open) or next_open <= 0:
                    continue
                
                # 计算最大可用资金（扣除佣金）
                max_cost = capital / (1 + self.commission_rate)
                shares = int(max_cost / next_open)
                
                if shares > 0:
                    cost = shares * next_open
                    commission = cost * self.commission_rate
                    
                    # 再次检查资金（双重保险）
                    if capital >= cost + commission:
                        # 计算买入时的筹码分布指标
                        # 截取截至当日的历史数据
                        vp_metrics = calculate_volume_profile_metrics(
                            closes[:i+1], 
                            self.df['Volume'].values[:i+1], 
                            next_open
                        )
                        
                        capital -= (cost + commission)
                        position = shares
                        entry_price = next_open
                        self.trades.append({
                            'type': 'BUY',
                            'date': dates[i+1],
                            'price': next_open,
                            'shares': shares,
                            'value': cost,
                            'reason': 'Heima/Juedi+DualBlue',
                            'vp_metrics': vp_metrics
                        })
            
            elif signal == -1:
                # 清仓卖出
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
                    'reason': 'KDJ_J>100'
                })
                position = 0
            
            # 记录当日市值
            current_value = capital + position * close_price
            portfolio_values.append(current_value)
            
        self.df['Portfolio'] = pd.Series(portfolio_values, index=self.df.index[:len(portfolio_values)])
        
        # 计算绩效
        final_value = portfolio_values[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        annual_return = (1 + total_return) ** (365 / self.days) - 1
        
        # 计算最大回撤
        portfolio_series = pd.Series(portfolio_values)
        rolling_max = portfolio_series.cummax()
        drawdown = (portfolio_series - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # 胜率
        winning_trades = len([t for t in self.trades if t.get('type') == 'SELL' and t.get('pnl', 0) > 0])
        total_closed_trades = len([t for t in self.trades if t.get('type') == 'SELL'])
        win_rate = winning_trades / total_closed_trades if total_closed_trades > 0 else 0
        
        self.results = {
            'Initial Capital': self.initial_capital,
            'Final Value': final_value,
            'Total Return': total_return,
            'Annual Return': annual_return,
            'Max Drawdown': max_drawdown,
            'Total Trades': total_closed_trades,
            'Win Rate': win_rate
        }
        
    def plot_results(self, show=True):
        if self.df is None or 'Portfolio' not in self.df.columns:
            return None
            
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                           vertical_spacing=0.03, subplot_titles=('Price & Signals', 'Portfolio Value'),
                           row_heights=[0.7, 0.3])

        # 1. 价格 K 线
        fig.add_trace(go.Candlestick(x=self.df.index,
                        open=self.df['Open'],
                        high=self.df['High'],
                        low=self.df['Low'],
                        close=self.df['Close'],
                        name='Price'), row=1, col=1)

        # 2. 买卖点标记
        buy_dates = [t['date'] for t in self.trades if t['type'] == 'BUY']
        buy_prices = [t['price'] for t in self.trades if t['type'] == 'BUY']
        sell_dates = [t['date'] for t in self.trades if t['type'] == 'SELL']
        sell_prices = [t['price'] for t in self.trades if t['type'] == 'SELL']
        
        fig.add_trace(go.Scatter(x=buy_dates, y=buy_prices, mode='markers', 
                                marker=dict(symbol='triangle-up', size=12, color='green', line=dict(width=1, color='black')),
                                name='Buy'), row=1, col=1)
                                
        fig.add_trace(go.Scatter(x=sell_dates, y=sell_prices, mode='markers', 
                                marker=dict(symbol='triangle-down', size=12, color='red', line=dict(width=1, color='black')),
                                name='Sell'), row=1, col=1)
                                
        # 3. 资金曲线
        fig.add_trace(go.Scatter(x=self.df.index, y=self.df['Portfolio'], 
                                mode='lines', name='Portfolio Value', line=dict(color='purple', width=2)), row=2, col=1)
        
        title = f"Backtest Result: {self.symbol} ({self.market})<br>"
        title += f"Return: {self.results['Total Return']:.2%} | Max DD: {self.results['Max Drawdown']:.2%} | Win Rate: {self.results['Win Rate']:.2%}"
        
        fig.update_layout(title=title, xaxis_rangeslider_visible=False, height=800)
        
        if show:
            fig.show()
            
        return fig

def main():
    parser = argparse.ArgumentParser(description='Simple Backtest for BLUE Strategy')
    parser.add_argument('symbol', type=str, help='Stock Symbol (e.g. AAPL, 600519)')
    parser.add_argument('--market', type=str, default='US', choices=['US', 'CN'], help='Market (US or CN)')
    parser.add_argument('--days', type=int, default=1095, help='Days to backtest (default 1095 = 3 years)')
    parser.add_argument('--capital', type=float, default=100000, help='Initial Capital')
    parser.add_argument('--threshold', type=float, default=100, help='BLUE signal threshold (default 100)')
    parser.add_argument('--heima', action='store_true', help='Require Heima/Juedi signal')
    parser.add_argument('--week_blue', action='store_true', help='Require Weekly BLUE signal')
    parser.add_argument('--vp_filter', action='store_true', help='Require good Volume Profile')
    
    args = parser.parse_args()
    
    backtester = SimpleBacktester(
        args.symbol, args.market, args.capital, args.days, 
        blue_threshold=args.threshold,
        require_heima=args.heima,
        require_week_blue=args.week_blue,
        require_vp_filter=args.vp_filter
    )
    if backtester.load_data():
        backtester.calculate_signals()
        backtester.run_backtest()
        
        print("\n" + "="*40)
        print(f"Backtest Results for {args.symbol}")
        print("="*40)
        for k, v in backtester.results.items():
            if 'Return' in k or 'Drawdown' in k or 'Rate' in k:
                print(f"{k:<20}: {v:.2%}")
            elif 'Value' in k or 'Capital' in k:
                print(f"{k:<20}: {v:,.2f}")
            else:
                print(f"{k:<20}: {v}")
        print("="*40)
        
        print("\nTrade Log:")
        print("-" * 100)
        print(f"{'Date':<12} {'Type':<6} {'Price':<10} {'Shares':<8} {'Profit':<10} {'VP_Profit%':<10} {'VP_Pos':<8} {'VP_Conc':<8}")
        print("-" * 100)
        for t in backtester.trades:
            pnl_str = f"{t['pnl']:.2f}" if 'pnl' in t else "-"
            vp = t.get('vp_metrics', {})
            profit_ratio = f"{vp.get('profit_ratio', 0):.2%}" if vp else "-"
            pos = vp.get('price_pos', '-')
            conc = f"{vp.get('concentration', 0):.2f}" if vp else "-"
            
            print(f"{t['date'].strftime('%Y-%m-%d'):<12} {t['type']:<6} {t['price']:<10.2f} {t['shares']:<8} {pnl_str:<10} {profit_ratio:<10} {pos:<8} {conc:<8}")
        print("-" * 100)
        
        backtester.plot_results()

if __name__ == "__main__":
    main()

