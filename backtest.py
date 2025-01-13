import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


class SignalBacktest:
    def __init__(self):
        # 信号分类
        self.UP_SIGNALS = {
            'smile_long_daily': '日PINK上穿10',
            'smile_long_weekly': '周PINK上穿10',
            'gold_vol_count': '黄金柱',
            'double_vol_count': '倍量柱',
            '零轴下金叉': 'MACD零轴下金叉',
            '底背离': 'MACD底背离'
        }

        self.DOWN_SIGNALS = {
            'smile_short_daily': '日PINK下穿94',
            'smile_short_weekly': '周PINK下穿94',
            '零轴上死叉': 'MACD零轴上死叉',
            '顶背离': 'MACD顶背离'
        }

        self.STRENGTH_SIGNALS = {
            'blue_days': '日BLUE>150',
            'blue_weeks': '周BLUE>150'
        }

    def load_data(self, file_path):
        """加载数据"""
        try:
            df = pd.read_csv(file_path)
            print(f"成功加载数据，共 {len(df)} 条记录")
            return df
        except Exception as e:
            print(f"加载数据失败: {e}")
            return None

    def analyze_single_signal(self, df, signal_name, signal_type='up'):
        """分析单个信号的表现"""
        if signal_type == 'up':
            success = lambda x: x > 0  # 价格上涨为成功
        else:
            success = lambda x: x < 0  # 价格下跌为成功

        # 对于计数类信号（如gold_vol_count）
        if signal_name in ['gold_vol_count', 'double_vol_count']:
            signal_df = df[df[signal_name] > 0]
        # 对于强度信号
        elif signal_name in ['blue_days', 'blue_weeks']:
            signal_df = df[df[signal_name] >= 3]  # 假设持续3天/周以上有效
        else:
            signal_df = df[df[signal_name] == True]

        if len(signal_df) == 0:
            return None

        return {
            'signal_name': self.UP_SIGNALS.get(signal_name) or self.DOWN_SIGNALS.get(
                signal_name) or self.STRENGTH_SIGNALS.get(signal_name),
            'count': len(signal_df),
            'success_rate': (signal_df['price_change_pct'].apply(success)).mean() * 100,
            'avg_return': signal_df['price_change_pct'].mean(),
            'max_gain': signal_df['max_gain_pct'].mean(),
            'max_drawdown': signal_df['max_drawdown_pct'].mean(),
            'avg_volume_change': signal_df['volume_change_pct'].mean()
        }

    def generate_labels(self, df):
        """为每个信号生成标签"""
        labels = pd.DataFrame(index=df.index)

        # 看涨信号标签
        for signal in self.UP_SIGNALS:
            if signal in ['gold_vol_count', 'double_vol_count']:
                labels[f'{signal}_label'] = np.where(
                    df[signal] > 0,
                    np.where(df['price_change_pct'] > 0, 1, 0),
                    None
                )
            else:
                labels[f'{signal}_label'] = np.where(
                    df[signal] == True,
                    np.where(df['price_change_pct'] > 0, 1, 0),
                    None
                )

        # 看跌信号标签
        for signal in self.DOWN_SIGNALS:
            labels[f'{signal}_label'] = np.where(
                df[signal] == True,
                np.where(df['price_change_pct'] < 0, 1, 0),
                None
            )

        # 强度信号标签
        for signal in self.STRENGTH_SIGNALS:
            labels[f'{signal}_label'] = np.where(
                df[signal] >= 3,
                np.where(df['price_change_pct'] > 0, 1, 0),
                None
            )

        return labels

    def run_analysis(self, df):
        """执行回测分析"""
        results = []

        # 分析做多信号
        for signal in self.UP_SIGNALS:
            result = self.analyze_single_signal(df, signal, 'up')
            if result:
                results.append(result)

        # 分析做空信号
        for signal in self.DOWN_SIGNALS:
            result = self.analyze_single_signal(df, signal, 'down')
            if result:
                results.append(result)

        # 分析强度信号
        for signal in self.STRENGTH_SIGNALS:
            result = self.analyze_single_signal(df, signal, 'up')
            if result:
                results.append(result)

        return pd.DataFrame(results)

    def save_results(self, results, labels, df, base_filename):
        """保存分析结果"""
        # 保存信号分析结果
        results.to_csv(f'{base_filename}_signal_performance.csv', index=False)
        print(f"信号表现分析已保存到 {base_filename}_signal_performance.csv")

        # 保存带标签的完整数据
        full_data = pd.concat([df, labels], axis=1)
        full_data.to_csv(f'{base_filename}_full_analysis.csv', index=False)
        print(f"完整分析数据已保存到 {base_filename}_full_analysis.csv")

    def create_summary_report(self, results):
        """创建汇总报告"""
        print("\n=== 信号表现汇总报告 ===")
        print("\n1. 信号出现频率:")
        print(results[['signal_name', 'count']].sort_values('count', ascending=False))

        print("\n2. 信号成功率:")
        print(results[['signal_name', 'success_rate']].sort_values('success_rate', ascending=False))

        print("\n3. 平均收益率:")
        print(results[['signal_name', 'avg_return']].sort_values('avg_return', ascending=False))

        print("\n4. 风险收益比:")
        results['risk_reward_ratio'] = abs(results['max_gain'] / results['max_drawdown'])
        print(results[['signal_name', 'risk_reward_ratio']].sort_values('risk_reward_ratio', ascending=False))


def main():
    # 初始化回测器
    backtest = SignalBacktest()

    # 加载数据
    df = backtest.load_data('signals_backtest_20250113_070324.csv')
    if df is None:
        return

    # 生成标签
    labels = backtest.generate_labels(df)

    # 执行分析
    results = backtest.run_analysis(df)

    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backtest.save_results(results, labels, df, f'backtest_results_{timestamp}')

    # 创建报告
    backtest.create_summary_report(results)


if __name__ == "__main__":
    main()