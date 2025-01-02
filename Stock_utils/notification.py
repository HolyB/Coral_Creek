import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import pandas as pd
import numpy as np

class SignalNotifier:
    def __init__(self, df, ticker: str):
        self.df = df
        self.ticker = ticker


    def send_resonance_notification(self):
        sender_email = "stockprofile138@gmail.com"
        receiver_emails = ["stockprofile138@gmail.com", "xhemobile@gmail.com"]  # 您的收件人列表
        subject = f"Bollinger Band Resonance Detected for {self.ticker}"

        body = f"Bollinger Band resonance has been detected for {self.ticker}.\n"
        body += f"Signal Type: {self.signal_type}\n\n"
        body += "Signals across intervals:\n"
        for interval, signal in self.signals_across_intervals.items():
            body += f"- Interval {interval}: {signal}\n"

        # 设置邮件内容
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        msg['To'] = ", ".join(receiver_emails)

        try:
            server = smtplib.SMTP('smtp.gmail.com', 587)  # 使用 Gmail SMTP 服务器
            server.starttls()
            server.login(sender_email, "your_email_password")  # 替换为您的邮箱密码或应用专用密码
            server.sendmail(sender_email, receiver_emails, msg.as_string())
            server.quit()
            print(f"Resonance notification email sent successfully for {self.ticker}.")
        except Exception as e:
            print(f"Failed to send resonance notification email for {self.ticker}: {e}")


    # def check_conditions(self):
    #     """
    #     检查最近45天内指定的信号中是否至少有三个出现过，
    #     并且当前价格没有比最近一次的'探底点'价格高出超过5%。
    #     """
    #     # 定义需要检查的信号
    #     signals_to_check = ['AI红柱', '火焰山底', '探底点', '爆量寻牛点']

    #     # 取最近45天的数据
    #     recent_df = self.df.tail(45)

    #     # 用于计数满足条件的信号数量
    #     signals_occurred_count = 0

    #     # 检查 '爆量' 大于30的情况，作为一个单独的信号
    #     if '爆量' in recent_df.columns:
    #         if (recent_df['爆量'] > 30).any():
    #             signals_occurred_count += 1
    #     else:
    #         print(f"'爆量' 列不存在于数据中。")

    #     # 检查每个信号是否在最近45天内出现过
    #     for signal in signals_to_check:
    #         if signal in recent_df.columns:
    #             if recent_df[signal].any():
    #                 signals_occurred_count += 1
    #         else:
    #             print(f"'{signal}' 列不存在于数据中。")

    #     # 检查 '探底点' 信号是否出现过
    #     if '探底点' in recent_df.columns and recent_df['探底点'].any():
    #         # 获取 '探底点' 信号最后一次出现的日期和对应的价格
    #         last_tandi_point_date = recent_df[recent_df['探底点'] != 0].index.max()
    #         tandi_price = self.df.loc[last_tandi_point_date, 'Close']

    #         # 获取当前价格
    #         current_price = self.df['Close'].iloc[-1]

    #         # 检查 tandi_price 是否为零
    #         if tandi_price == 0:
    #             print("探底点价格为零，无法计算涨幅。不发送邮件。")
    #             return False

    #         # 检查当前价格是否比 '探底点' 价格高出超过5%
    #         price_increase = (current_price - tandi_price) / tandi_price
    #         if price_increase > 0.05:
    #             print(f"当前价格比'探底点'价格高出超过5%（涨幅{price_increase:.2%}）。不发送邮件。")
    #             return False
    #     else:
    #         # 如果 '探底点' 没有出现过，或者列不存在，则不发送邮件
    #         print(f"最近45天内未出现'探底点'信号。不发送邮件。")
    #         return False

    #     # 如果满足条件的信号数量大于等于3，返回 True，否则返回 False
    #     return signals_occurred_count >= 3



    def check_conditions(self):
        """
        检查最近45天内指定的信号中是否至少有三个出现过，
        并且当前价格没有比最近一次的'探底点'价格高出超过5%。
        满足条件时，保存信号信息以供后续使用。
        返回布尔值 True（满足条件）或 False（不满足条件）。
        """
        print("*********************************************************************")
        # 定义需要检查的信号
        signals_to_check = ['AI红柱', '火焰山底', '探底点', '爆量寻牛点']

        # 取最近45天的数据
        recent_df = self.df.tail(45)

        # 用于存储满足条件的信号
        self.occurred_signals = []

        # 检查 '爆量' 大于30的情况，作为一个单独的信号
        if '爆量' in recent_df.columns:
            if (recent_df['爆量'] > 30).any():
                self.occurred_signals.append('爆量>30')
        else:
            print(f"'爆量' 列不存在于数据中。")

        # 检查每个信号是否在最近45天内出现过
        for signal in signals_to_check:
            if signal in recent_df.columns:
                if recent_df[signal].any():
                    self.occurred_signals.append(signal)
            else:
                print(f"'{signal}' 列不存在于数据中。")

        # 检查是否至少有三个信号出现过
        if len(self.occurred_signals) < 3:
            return False  # 不满足条件

        # 检查 '探底点' 信号是否出现过
        if '探底点' in recent_df.columns and recent_df['探底点'].any():
            # 获取 '探底点' 信号最后一次出现的日期和对应的价格
            last_tandi_point_date = recent_df[recent_df['探底点'] != 0].index.max()
            tandi_price = self.df.loc[last_tandi_point_date, 'Close']

            # 获取当前价格
            current_price = self.df['Close'].iloc[-1]

            # 检查 tandi_price 是否为零
            if tandi_price == 0:
                print("探底点价格为零，无法计算涨幅。不发送邮件。")
                return False

            # 检查当前价格是否比 '探底点' 价格高出超过5%
            price_increase = (current_price - tandi_price) / tandi_price
            if price_increase > 0.5:
                print(f"当前价格比'探底点'价格高出超过5%（涨幅{price_increase:.2%}）。不发送邮件。")
                return False

            # 保存相关信息以供后续使用
            self.tandi_price = tandi_price
            self.current_price = current_price
            self.price_increase = price_increase
        else:
            # 如果 '探底点' 没有出现过，或者列不存在，则不发送邮件
            print(f"最近45天内未出现'探底点'信号。不发送邮件。")
            return False

        # 如果所有条件都满足，返回 True
        return True


    def check_dt_conditions(self): 
        """
        检查最近45天内指定的信号中是否至少有三个出现过，
        并且当前价格没有比最近一次的'探底点'或其他高点信号价格高出超过5%。
        满足条件时，保存信号信息以供后续使用。
        返回布尔值 True（满足条件）或 False（不满足条件）。
        """
        print("*********************************************************************")
        # 定义需要检查的信号
        signals_to_check = ['AI红柱', '火焰山底', '探底点', '爆量寻牛点', 'AI绿柱', '趋势空', '启动上涨', '逃顶']

        # 取最近8天的数据
        recent_df = self.df.tail(8)

        # 用于存储满足条件的信号
        self.occurred_signals = []

        # 检查 '爆量' 大于30的情况，作为一个单独的信号
        if '爆量' in recent_df.columns:
            if (recent_df['爆量'] > 30).any():
                self.occurred_signals.append('爆量>30')
        else:
            print(f"'爆量' 列不存在于数据中。")

        # 检查每个信号是否在最近45天内出现过
        for signal in signals_to_check:
            if signal in recent_df.columns:
                if recent_df[signal].any():
                    self.occurred_signals.append(signal)
            else:
                print(f"'{signal}' 列不存在于数据中。")

        # 检查是否至少有三个信号出现过
        if len(self.occurred_signals) < 3:
            return False  # 不满足条件

        # 检查 '探底点' 或者其他高点信号（如'启动上涨'、'逃顶'）是否出现过
        key_signals = ['探底点', '启动上涨', '逃顶', '趋势空']
        price_checked = False  # 标记是否检查了价格
        current_price = self.df['Close'].iloc[-1]
        self.current_price = current_price
        # for key_signal in key_signals:
        #     if key_signal in recent_df.columns and recent_df[key_signal].any():
        #         # 获取该信号最后一次出现的日期和对应的价格
        #         last_signal_date = recent_df[recent_df[key_signal] != 0].index.max()
        #         signal_price = self.df.loc[last_signal_date, 'Close']

        #         # 获取当前价格
        #         current_price = self.df['Close'].iloc[-1]

        #         # 检查 signal_price 是否为零
        #         if signal_price == 0:
        #             print(f"{key_signal}价格为零，无法计算涨幅。不发送邮件。")
        #             return False

        #         # 检查当前价格是否比该信号价格高出超过5%
        #         price_increase = (current_price - signal_price) / signal_price
        #         if price_increase > 0.05:
        #             print(f"当前价格比'{key_signal}'价格高出超过5%（涨幅{price_increase:.2%}）。不发送邮件。")
        #             return False

        #         # 保存相关信息以供后续使用
        #         self.signal_price = signal_price
        #         self.current_price = current_price
        #         self.price_increase = price_increase
        #         price_checked = True
        #         break  # 已找到符合条件的信号，不再继续检查其他关键信号

        # if not price_checked:
        #     # 如果没有任何关键信号满足条件，则不发送邮件
        #     print("最近45天内未出现任何'探底点'或高点信号。不发送邮件。")
        #     return False

        # 如果所有条件都满足，返回 True
        return True


    def summarize_signals(self):
        # Define the main signals to check
        signals = ['AI红柱', 'AI绿柱','主力出没', '洗盘', '底背离', '顶背离', 'brown_底背离', 'white_底背离', 'green_顶背离', '火焰山底',
                   '探底点', '爆量', '爆量寻牛点', '能量悬性', '能量买卖信号', '主力持仓', '散户持仓', '黄钻', '绿钻', '资金进出趋势',
                   '大底', '是底', '逃顶', '启动上涨', '趋势多', '趋势空', '冲刺红柱', '下降白柱', '红箭头', '绿箭头']
        
        summary = {}
        recent_signals = []
        for signal in signals:
            if signal in self.df.columns:
                count_5days = self.df[signal].rolling(window=5, min_periods=1).sum().iloc[-1]
                count_15days = self.df[signal].rolling(window=15, min_periods=1).sum().iloc[-1]
                count_30days = self.df[signal].rolling(window=30, min_periods=1).sum().iloc[-1]
                count_90days = self.df[signal].rolling(window=90, min_periods=1).sum().iloc[-1]
                last_occurrence = self.df[self.df[signal].astype(bool)].index.max()
                last_price = self.df['Close'][last_occurrence] if pd.notna(last_occurrence) else None
                
                summary[signal] = {
                    'count_5days': int(count_5days) if not np.isnan(count_5days) else 0,
                    'count_15days': int(count_15days) if not np.isnan(count_15days) else 0,
                    'count_30days': int(count_30days) if not np.isnan(count_30days) else 0,
                    'count_90days': int(count_90days) if not np.isnan(count_90days) else 0,
                    'last_price': round(last_price, 2) if last_price is not None else "N/A",
                    'last_occurrence': last_occurrence if pd.notna(last_occurrence) else 'N/A'
                }
                # Add to recent signals if any count is greater than zero
                if any([summary[signal]['count_5days'], summary[signal]['count_15days'], summary[signal]['count_30days'], summary[signal]['count_90days']]):
                    recent_signals.append(signal)
        
        return summary, recent_signals

    def send_zhuli_xipan_email(self, signal_stats):
        sender_email = "stockprofile138@gmail.com"
        receiver_emails = ["stockprofile138@gmail.com", "xhemobile@gmail.com"]  # 您的收件人列表
        subject = f"'{self.ticker}' 的 '主力出没' 和 '洗盘' 信号报告"

        body = f"在最近 20 个 interval 内，'{self.ticker}' 的 '主力出没' 和 '洗盘' 信号统计如下：\n\n"

        for interval, signals in signal_stats.items():
            body += f"时间周期: {interval}\n"
            for signal_name in ['主力出没', '洗盘']:
                signal_info = signals.get(signal_name)
                if signal_info:
                    body += f"  信号: {signal_name}\n"
                    body += f"    出现次数: {signal_info['num_occurrences']}\n"
                    body += f"    最近一次值: {signal_info['latest_value']}（日期: {signal_info['latest_date']}）\n"
                    body += f"    最大值: {signal_info['max_value']}（日期: {signal_info['max_date']}）\n"
                    body += f"    最大值对应的股票价格: {signal_info['max_price']}\n"
                else:
                    body += f"  信号: {signal_name}\n"
                    body += f"    最近 20 个 interval 内未出现。\n"
            body += "\n"

        body += "请对该股票进行进一步分析，以寻找潜在的交易机会。"

        # 设置邮件内容
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain', 'utf-8'))
        msg['To'] = ", ".join(receiver_emails)

        try:
            server = smtplib.SMTP('smtp.gmail.com', 587)  # 使用 Gmail SMTP 服务器
            server.starttls()
            server.login(sender_email, "vselpmwrjacmgdib")  # 替换为您的邮箱密码或应用专用密码
            server.sendmail(sender_email, receiver_emails, msg.as_string())
            server.quit()
            print(f"Email sent successfully for {self.ticker}.")
        except Exception as e:
            print(f"Failed to send email for {self.ticker}: {e}")

    def calculate_signal_stats(self):
        """
        计算信号统计信息，返回 signal_stats 字典。
        """
        signal_stats = {}
        # 假设 self.df 中包含 interval 信息，如果没有，需要根据实际情况调整
        # 这里的 interval 可以是一个列，或者其他方式标识不同的时间周期

        # 因为 self.df 只包含当前 interval 的数据，我们可以直接计算
        interval = "current_interval"  # 根据实际情况修改

        signal_stats[interval] = {}
        signals_to_check = ['主力出没', '洗盘']

        for signal in signals_to_check:
            if signal in self.df.columns:
                # 获取最近 20 个 interval 的信号值
                signal_last_20 = self.df[signal].tail(20)

                # 出现次数
                num_occurrences = (signal_last_20 > 0).sum()

                if num_occurrences > 0:
                    # 最近一次的值和日期
                    recent_values = signal_last_20[signal_last_20 > 0]
                    latest_value = recent_values.iloc[-1]
                    latest_date = recent_values.index[-1]

                    # 最大值及其出现的日期
                    max_value = recent_values.max()
                    max_date = recent_values.idxmax()

                    # 最大值对应的股票价格
                    max_price = self.df.loc[max_date, 'close']

                    # 存储信号统计信息
                    signal_stats[interval][signal] = {
                        'num_occurrences': num_occurrences,
                        'latest_value': latest_value,
                        'latest_date': latest_date,
                        'max_value': max_value,
                        'max_date': max_date,
                        'max_price': max_price
                    }
                else:
                    # 最近 20 个 interval 内未出现该信号
                    signal_stats[interval][signal] = None
            else:
                print(f"Signal '{signal}' not found in data for interval {interval}")
                signal_stats[interval][signal] = None

        return signal_stats

    def send_summary_email(self):
        # 在发送邮件之前，先检查条件
        if not self.check_conditions():
            print(f"No significant signals for {self.ticker}. Email not sent.")
            return

        # 使用在 check_conditions 中保存的信息
        occurred_signals = self.occurred_signals
        tandi_price = self.tandi_price
        current_price = self.current_price
        price_increase = self.price_increase

        # Summarize the signals
        summary, recent_signals = self.summarize_signals()
        current_volume = self.df['Volume'].iloc[-1]
        moving_avg_5d = self.df['Close'].rolling(window=5).mean().iloc[-1]
        moving_avg_volume_5d = self.df['Volume'].rolling(window=5).mean().iloc[-1]

        # 获取最近15天内出现过的信号
        signals_in_last_15_days = []
        signals = ['AI红柱', 'AI绿柱', '主力出没', '洗盘', '底背离', '顶背离', 'brown_底背离', 'white_底背离', 'green_顶背离', '火焰山底',
                  '探底点', '爆量', '爆量寻牛点', '能量悬性', '能量买卖信号', '主力持仓', '散户持仓', '黄钻', '绿钻', '资金进出趋势',
                  '大底', '是底', '逃顶', '启动上涨', '趋势多', '趋势空', '冲刺红柱', '下降白柱', '红箭头', '绿箭头']

        recent_15_days_df = self.df.tail(15)

        for signal in signals:
            if signal in recent_15_days_df.columns:
                if recent_15_days_df[signal].any():
                    signals_in_last_15_days.append(signal)

        # 将最近15天内的信号摘要添加到邮件标题中
        if signals_in_last_15_days:
            signals_summary = ", ".join(signals_in_last_15_days)
        else:
            signals_summary = "No recent signals"

        sender_email = "stockprofile138@gmail.com"
        receiver_emails = ["stockprofile138@gmail.com", "xhemobile@gmail.com"]  # 您的收件人列表
        subject = f"Trading Summary for {self.ticker} - Recent Signals: {occurred_signals}"

        body = f"Summary of detected trading opportunities for {self.ticker}:\n"
        body += f"Current Price: {current_price}\nCurrent Volume: {current_volume}\n"
        body += f"5-Day Moving Average Price: {moving_avg_5d}\n5-Day Moving Average Volume: {moving_avg_volume_5d}\n\n"

        body += f"满足条件的信号：{', '.join(occurred_signals)}\n"
        body += f"探底点价格：{tandi_price}\n"
        body += f"当前价格比探底点价格上涨了：{price_increase:.2%}\n\n"

        if recent_signals:
            body += "Recently detected signals: " + ", ".join(recent_signals) + "\n\n"
        else:
            body += "No recent signals detected.\n\n"

        body += "Signals Summary:\n"
        for signal, details in summary.items():
            body += f"{signal}:\n"
            body += f"  - Last 5 days: {details['count_5days']} times\n"
            body += f"  - Last 15 days: {details['count_15days']} times\n"
            body += f"  - Last 30 days: {details['count_30days']} times\n"
            body += f"  - Last 90 days: {details['count_90days']} times\n"
            body += f"  - Last observed price: {details['last_price']}\n"
            body += f"  - Last occurrence: {details['last_occurrence']}\n"

        # 添加机器学习模型的预测概率到邮件正文
        if 'UpDown_Prob' in self.df.columns and 'BuyPoint_Prob' in self.df.columns:
            body += f"\nUp/Down Probability (Random Forest) for the latest data: {self.df['UpDown_Prob'].iloc[-1]:.2f}\n"
            body += f"Buy Point Probability (Random Forest) for the latest data: {self.df['BuyPoint_Prob'].iloc[-1]:.2f}\n"
        if 'ARIMA_Prediction' in self.df.columns:
            body += f"ARIMA Prediction for the latest data: {self.df['ARIMA_Prediction'].iloc[-1]:.2f}\n"

        if 'XGBoost_Prediction' in self.df.columns:
            body += f"XGBoost Prediction for the latest data: {self.df['XGBoost_Prediction'].iloc[-1]:.2f}\n"

        if 'LSTM_Prediction' in self.df.columns:
            body += f"LSTM Prediction for the latest data: {self.df['LSTM_Prediction'].iloc[-1]:.2f}\n"

        # 设置邮件内容
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        msg['Bcc'] = ", ".join(receiver_emails)

        try:
            server = smtplib.SMTP('smtp.gmail.com', 587)  # 使用 Gmail SMTP 服务器
            server.starttls()
            server.login(sender_email, "vselpmwrjacmgdib")  # 替换为您的邮箱密码或应用专用密码
            server.sendmail(sender_email, receiver_emails, msg.as_string())
            server.quit()
            print(f"Summary email sent successfully for {self.ticker}.")
        except Exception as e:
            print(f"Failed to send summary email for {self.ticker}: {e}")


    def send_dt_summary_email(self, interval):
        # 在发送邮件之前，先检查条件
        if not self.check_dt_conditions():
            print(f"No significant signals for {self.ticker}. Email not sent.")
            return

        # 使用在 check_conditions 中保存的信息
        occurred_signals = self.occurred_signals
        current_price = self.current_price

        # Summarize the signals
        summary, recent_signals = self.summarize_signals()
        current_volume = self.df['Volume'].iloc[-1]
        moving_avg_5d = self.df['Close'].rolling(window=5).mean().iloc[-1]
        moving_avg_volume_5d = self.df['Volume'].rolling(window=5).mean().iloc[-1]

        # 获取最近15天内出现过的信号
        signals_in_last_15_days = []
        signals = ['AI红柱', 'AI绿柱', '主力出没', '洗盘', '底背离', '顶背离', 'brown_底背离', 'white_底背离', 
                  'green_顶背离', '火焰山底', '探底点', '爆量', '爆量寻牛点', '能量悬性', '能量买卖信号', 
                  '主力持仓', '散户持仓', '黄钻', '绿钻', '资金进出趋势', '大底', '是底', '逃顶', '启动上涨', 
                  '趋势多', '趋势空', '冲刺红柱', '下降白柱', '红箭头', '绿箭头','趋势空', '启动上涨','逃顶']

        recent_15_days_df = self.df.tail(15)

        for signal in signals:
            if signal in recent_15_days_df.columns and recent_15_days_df[signal].any():
                signals_in_last_15_days.append(signal)

        # 将最近15天内的信号摘要添加到邮件标题中
        if signals_in_last_15_days:
            signals_summary = ", ".join(signals_in_last_15_days)
        else:
            signals_summary = "No recent signals"

        sender_email = "stockprofile138@gmail.com"
        receiver_emails = ["stockprofile138@gmail.com", "xhemobile@gmail.com"]  # 您的收件人列表
        subject = f"Trading Summary for {self.ticker} - {interval} Interval - Recent Signals: {occurred_signals}"

        body = f"Summary of detected trading opportunities for {self.ticker}:\n"
        body += f"Current Price: {current_price}\nCurrent Volume: {current_volume}\n"
        body += f"5-Day Moving Average Price: {moving_avg_5d}\n5-Day Moving Average Volume: {moving_avg_volume_5d}\n\n"

        body += f"满足条件的信号：{', '.join(occurred_signals)}\n"

        if recent_signals:
            body += "Recently detected signals: " + ", ".join(recent_signals) + "\n\n"
        else:
            body += "No recent signals detected.\n\n"

        body += "Signals Summary:\n"
        for signal, details in summary.items():
            body += f"{signal}:\n"
            body += f"  - Last 5 days: {details['count_5days']} times\n"
            body += f"  - Last 15 days: {details['count_15days']} times\n"
            body += f"  - Last 30 days: {details['count_30days']} times\n"
            body += f"  - Last 90 days: {details['count_90days']} times\n"
            body += f"  - Last observed price: {details['last_price']}\n"
            body += f"  - Last occurrence: {details['last_occurrence']}\n"

        # 设置邮件内容
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        msg['Bcc'] = ", ".join(receiver_emails)

        try:
            server = smtplib.SMTP('smtp.gmail.com', 587)  # 使用 Gmail SMTP 服务器
            server.starttls()
            server.login(sender_email, "vselpmwrjacmgdib")  # 替换为您的邮箱密码或应用专用密码
            server.sendmail(sender_email, receiver_emails, msg.as_string())
            server.quit()
            print(f"Summary email sent successfully for {self.ticker} at {interval} interval.")
        except Exception as e:
            print(f"Failed to send summary email for {self.ticker}: {e}")


    # def send_summary_email(self):
    #     # 在发送邮件之前，先检查条件
    #     if not self.check_conditions():
    #         print(f"No significant signals for {self.ticker} in the last 60 days. Email not sent.")
    #         return

    #     # Summarize the signals
    #     summary, recent_signals = self.summarize_signals()
    #     current_price = self.df['Close'].iloc[-1]
    #     current_volume = self.df['Volume'].iloc[-1]
    #     moving_avg_5d = self.df['Close'].rolling(window=5).mean().iloc[-1]
    #     moving_avg_volume_5d = self.df['Volume'].rolling(window=5).mean().iloc[-1]

    #     # **新增：获取最近15天内出现过的信号**
    #     signals_in_last_15_days = []
    #     signals = ['AI红柱', 'AI绿柱', '主力出没', '洗盘', '底背离', '顶背离', 'brown_底背离', 'white_底背离', 'green_顶背离', '火焰山底',
    #               '探底点', '爆量', '爆量寻牛点', '能量悬性', '能量买卖信号', '主力持仓', '散户持仓', '黄钻', '绿钻', '资金进出趋势',
    #               '大底', '是底', '逃顶', '启动上涨', '趋势多', '趋势空', '冲刺红柱', '下降白柱', '红箭头', '绿箭头']

    #     recent_15_days_df = self.df.tail(15)

    #     for signal in signals:
    #         if signal in recent_15_days_df.columns:
    #             if recent_15_days_df[signal].any():
    #                 signals_in_last_15_days.append(signal)

    #     # **将最近15天内的信号摘要添加到邮件标题中**
    #     if signals_in_last_15_days:
    #         signals_summary = ", ".join(signals_in_last_15_days)
    #     else:
    #         signals_summary = "No recent signals"

    #     sender_email = "stockprofile138@gmail.com"
    #     receiver_emails = ["stockprofile138@gmail.com", "xhemobile@gmail.com"]  # 您的收件人列表
    #     subject = f"Trading Summary for {self.ticker} - Recent Signals: {signals_summary}"

    #     body = f"Summary of detected trading opportunities for {self.ticker}:\n"
    #     body += f"Current Price: {current_price}\nCurrent Volume: {current_volume}\n"
    #     body += f"5-Day Moving Average Price: {moving_avg_5d}\n5-Day Moving Average Volume: {moving_avg_volume_5d}\n\n"
    #     if recent_signals:
    #         body += "Recently detected signals: " + ", ".join(recent_signals) + "\n\n"
    #     else:
    #         body += "No recent signals detected.\n\n"

    #     body += "Signals Summary:\n"
    #     for signal, details in summary.items():
    #         body += f"{signal}:\n"
    #         body += f"  - Last 5 days: {details['count_5days']} times\n"
    #         body += f"  - Last 15 days: {details['count_15days']} times\n"
    #         body += f"  - Last 30 days: {details['count_30days']} times\n"
    #         body += f"  - Last 90 days: {details['count_90days']} times\n"
    #         body += f"  - Last observed price: {details['last_price']}\n"
    #         body += f"  - Last occurrence: {details['last_occurrence']}\n"

    #     # Add prediction probabilities from Random Forest model to the email body
    #     if 'UpDown_Prob' in self.df.columns and 'BuyPoint_Prob' in self.df.columns:
    #         body += f"\nUp/Down Probability (Random Forest) for the latest data: {self.df['UpDown_Prob'].iloc[-1]:.2f}\n"
    #         body += f"Buy Point Probability (Random Forest) for the latest data: {self.df['BuyPoint_Prob'].iloc[-1]:.2f}\n"
    #     if 'ARIMA_Prediction' in self.df.columns:
    #         body += f"ARIMA Prediction for the latest data: {self.df['ARIMA_Prediction'].iloc[-1]:.2f}\n"

    #     if 'XGBoost_Prediction' in self.df.columns:
    #         body += f"XGBoost Prediction for the latest data: {self.df['XGBoost_Prediction'].iloc[-1]:.2f}\n"

    #     if 'LSTM_Prediction' in self.df.columns:
    #         body += f"LSTM Prediction for the latest data: {self.df['LSTM_Prediction'].iloc[-1]:.2f}\n"

    #     # Set up the MIME
    #     msg = MIMEMultipart()
    #     msg['From'] = sender_email
    #     msg['Subject'] = subject
    #     msg.attach(MIMEText(body, 'plain'))
    #     msg['Bcc'] = ", ".join(receiver_emails)

    #     try:
    #         server = smtplib.SMTP('smtp.gmail.com', 587)  # Using Gmail SMTP server
    #         server.starttls()
    #         server.login(sender_email, "vselpmwrjacmgdib")  # 替换为您的邮箱密码或应用专用密码
    #         server.sendmail(sender_email, receiver_emails, msg.as_string())
    #         server.quit()
    #         print(f"Summary email sent successfully for {self.ticker}.")
    #     except Exception as e:
    #         print(f"Failed to send summary email for {self.ticker}: {e}")



    def calculate_label_statistics(self):
        """
        计算前5天、前15天、前30天和前90天内的指标信号次数和最近一次信号对应价格。
        """
        signals = [
            'AI红柱', 'AI绿柱', 'brown_底背离', 'white_底背离', 'green_顶背离',
            '火焰山底', '探底点', '爆量寻牛点', '能量买卖信号', '黄钻', '绿钻',
            '大底', '底部扶摇直上', 'BU', 'SEL', '启动上涨', '红箭头', '绿箭头'
        ]
        periods = [5, 15, 30, 60, 90]

        for signal in signals:
            if signal in self.df.columns:
                for period in periods:
                    count_col = f'{signal}_count_last_{period}d'
                    price_col = f'{signal}_last_price_{period}d'

                    self.df[count_col] = self.df[signal].rolling(window=period, min_periods=1).apply(lambda x: np.nansum(~np.isnan(x)), raw=True)
                    self.df[price_col] = self.df[signal].rolling(window=period, min_periods=1).apply(lambda x: self.df['Close'].iloc[np.where(~np.isnan(x))[0][-1]] if len(np.where(~np.isnan(x))[0]) > 0 else np.nan, raw=True)



# Example usage
if __name__ == "__main__":
    # Sample dataframe with signal columns
    data = {
        '主力出没': [0, 1, 0, 0, 1, 0, 1],
        '洗盘': [1, 0, 1, 0, 0, 1, 0],
        '底背离': [0, 0, 1, 0, 0, 0, 1],
        'Close': [100, 102, 101, 105, 110, 115, 120],
        'Volume': [1000, 1500, 1200, 1300, 1700, 1600, 2000],
        'UpDown_Prob': [0.6, 0.7, 0.55, 0.65, 0.8, 0.75, 0.9],
        'BuyPoint_Prob': [0.4, 0.5, 0.3, 0.45, 0.6, 0.5, 0.7]
    }
    df = pd.DataFrame(data)
    df.index = pd.date_range(start='2023-01-01', periods=len(df))
    
    # Create an instance of SignalNotifier
    notifier = SignalNotifier(df, ticker='AAPL')
    
    # Calculate label statistics
    notifier.calculate_label_statistics()
    
    # Send a summary email
    notifier.send_summary_email()