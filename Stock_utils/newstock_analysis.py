import pandas as pd
import numpy as np
from datetime import datetime
from .MyTT import *



class StockAnalysis:
    def __init__(self, df: pd.DataFrame):
        """
        初始化 StockAnalysis 类，输入包含至少 'Open','High','Low','Close' 列的 DataFrame。
        :param df: 包含股票历史数据的 DataFrame
        """
        self.df = df.copy()
        # 确保列名标准化
        if 'open' in self.df.columns:
            self.df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 
                         'close': 'Close', 'volume': 'Volume'}, inplace=True)

    def calculate_phantom_indicators(self):
        """
        计算幻影主力指标，完全按照富途的通达信公式实现
        
        富途公式:
        {幻影主力=资金力度+海底捞月+KDJ}
        {海底捞月部分}
        VAR1:=REF((LOW+OPEN+CLOSE+HIGH)/4,1);
        VAR2:=SMA(ABS(LOW-VAR1),13,1)/SMA(MAX(LOW-VAR1,0),10,1);
        VAR3:=EMA(VAR2,10);
        VAR4:=LLV(LOW,33);
        VAR5:=EMA(IF(LOW <=VAR4,VAR3,0),3);
        VAR6:=POW(VAR5,0.3);
        {使用CONST确定缩放比例}
        RADIO1:=200/CONST(HHV(MAX(VAR6,VAR61),TOTALBARSCOUNT));
        BLUE:IF(VAR5 > REF(VAR5,1),VAR6*RADIO1,0),COLOR0000FF,STICK,LINETHICK5;{海底捞月}
        
        返回:
        添加了幻影主力指标的DataFrame
        """
        try:
            df = self.df.copy()
            
            # 提取OHLCV数据
            OPEN = np.array(df['Open'])
            HIGH = np.array(df['High'])
            LOW = np.array(df['Low'])
            CLOSE = np.array(df['Close'])
            if 'Volume' in df.columns:
                VOLUME = np.array(df['Volume'])
            else:
                VOLUME = np.zeros_like(CLOSE)
            
            # ------------------------ 海底捞月部分 ------------------------
            # VAR1:=REF((LOW+OPEN+CLOSE+HIGH)/4,1);
            VAR1 = REF((LOW + OPEN + CLOSE + HIGH) / 4, 1)
            
            # VAR2:=SMA(ABS(LOW-VAR1),13,1)/SMA(MAX(LOW-VAR1,0),10,1);
            VAR2 = SMA(ABS(LOW - VAR1), 13, 1) / SMA(MAX(LOW - VAR1, 0), 10, 1)
            
            # VAR3:=EMA(VAR2,10);
            VAR3 = EMA(VAR2, 10)
            
            # VAR4:=LLV(LOW,33);
            VAR4 = LLV(LOW, 9)
            
            # VAR5:=EMA(IF(LOW <=VAR4,VAR3,0),3);
            VAR5 = EMA(IF(LOW <= VAR4, VAR3, 0), 3)
            
            # VAR6:=POW(VAR5,0.3);
            VAR6 = POW(VAR5, 0.3)
            
            # 负向海底捞月
            # VAR21:=SMA(ABS(HIGH-VAR1),13,1)/SMA(MIN(HIGH-VAR1,0),10,1);
            VAR21 = SMA(ABS(HIGH - VAR1), 13, 1) / SMA(MIN(HIGH - VAR1, 0), 10, 1)
            
            # VAR31:=EMA(VAR21,10);
            VAR31 = EMA(VAR21, 10)
            
            # VAR41:=HHV(HIGH,33);
            VAR41 = HHV(HIGH, 9)
            
            # # VAR51:=EMA(IF(HIGH >=VAR41,-VAR31,0),3);
            # VAR51 = EMA(IF(HIGH >= VAR41, -VAR31, 0), 3)
            
            # # VAR61:=POW(VAR51,0.3);
            # VAR61 = POW(VAR51, 0.3)
            

            VAR51 = EMA(IF(HIGH >= VAR41, -VAR31, 0), 3)

            VAR61 = POW(VAR51, 0.3)
            # 使用缩放比例
            # RADIO1:=200/CONST(HHV(MAX(VAR6,VAR61),TOTALBARSCOUNT));
            # 注意：TOTALBARSCOUNT在通达信中是总K线数量，这里用len(HIGH)替代
            max_value = np.nanmax(np.maximum(VAR6, VAR61))  # 使用nanmax处理可能的nan值
            RADIO1 = 200 / max_value if max_value > 0 else 1
            
            # 计算BLUE和LIRED
            # BLUE:IF(VAR5 > REF(VAR5,1),VAR6*RADIO1,0)
            BLUE = IF(VAR5 > REF(VAR5, 1), VAR6 * RADIO1, 0)
            
            # LIRED:IF(VAR51 > REF(VAR51,1),-VAR61*RADIO1,0)
            # LIRED = IF(VAR51 > REF(VAR51, 1), -VAR61 * RADIO1, 0)
            LIRED = IF(VAR51 > REF(VAR51, 1), -VAR61 * RADIO1, 0)
            
            # ------------------------ 资金力度部分 ------------------------
            # QJJ:=VOLA / ( (H-L)*2-ABS(C-O) );
            QJJ = VOLUME / ((HIGH - LOW) * 2 - ABS(CLOSE - OPEN))
            # 处理除零情况
            QJJ = np.where(np.isnan(QJJ) | np.isinf(QJJ), 0, QJJ)
            
            # XVL:=IF(C=O,0,(C-O)*QJJ);
            XVL = IF(CLOSE == OPEN, 0, (CLOSE - OPEN) * QJJ)
            
            # HSL:=((XVL / 20) / 1.15);
            HSL = XVL / 20 / 1.15
            
            # 攻击流量:=HSL*0.55 + REF(HSL,1)*0.33 + REF(HSL,2)*0.22;
            攻击流量 = HSL * 0.55 + REF(HSL, 1) * 0.33 + REF(HSL, 2) * 0.22
            
            # LLJX:=EMA(攻击流量,3);
            LLJX = EMA(攻击流量, 3)
            
            # 使用缩放比例
            # RADIO:=10000/CONST(HHV(VOLA,TOTALBARSCOUNT));
            max_volume = np.nanmax(VOLUME) if len(VOLUME) > 0 else 1
            RADIO = 10000 / max_volume if max_volume > 0 else 10000
            
            # RED:IF(LLJX>0,LLJX,0),COLORFF0000,NODRAW;
            RED = IF(LLJX > 0, LLJX, 0)
            RED_STICK = RED * RADIO
            
            # YELLOW:IF(HSL>0,HSL*0.6,0),COLORFFFF00,NODRAW;
            YELLOW = IF(HSL > 0, HSL * 0.6, 0)
            YELLOW_STICK = YELLOW * RADIO
            
            # GREEN:IF(LLJX<0 OR HSL<0, MIN(LLJX,HSL*0.6),0),COLOR00FF00,NODRAW;
            GREEN = IF((LLJX < 0) | (HSL < 0), np.minimum(LLJX, HSL * 0.6), 0)
            GREEN_STICK = GREEN * RADIO
            
            # ------------------------ KDJ部分 ------------------------
            # RSV1:=(C-LLV(LOW,39))/(HHV(HIGH,39)-LLV(LOW,39))*100;
            RSV1 = (CLOSE - LLV(LOW, 39)) / (HHV(HIGH, 39) - LLV(LOW, 39)) * 100
            # 处理除零情况
            RSV1 = np.where(np.isnan(RSV1) | np.isinf(RSV1), 0, RSV1)
            
            # K:=SMA(RSV1,2,1);
            K = SMA(RSV1, 2, 1)
            
            # D:=SMA(K,2,1);
            D = SMA(K, 2, 1)
            
            # J:=3*K-2*D;
            J = 3 * K - 2 * D
            
            # PINK:SMA(J,2,1),COLORFF00FF;
            PINK = SMA(J, 2, 1)
            
            # LIGHTBLUE:DMA(攻击流量, 0.222228),COLOR00FFFF,NODRAW;
            LIGHTBLUE = DMA(攻击流量, 0.222228)
            LIGHTBLUE_LINE = LIGHTBLUE * RADIO
            
            # ------------------------ 笑脸信号 ------------------------
            # DRAWICON(CROSS(94,PINK),L*0.03,15); - 做空信号
            # DRAWICON(CROSS(PINK,10),H*0.03,5);  - 做多信号
            笑脸信号_做多 = np.zeros_like(PINK, dtype=int)
            笑脸信号_做空 = np.zeros_like(PINK, dtype=int)
            
            for i in range(1, len(PINK)):
                # 做多信号：PINK上穿10
                if PINK[i-1] <= 10 and PINK[i] > 10:
                    笑脸信号_做多[i] = 1
                    
                # 做空信号：PINK下穿94
                if PINK[i-1] >= 94 and PINK[i] < 94:
                    笑脸信号_做空[i] = 1
            
            # 将结果添加到DataFrame
            df['BLUE'] = BLUE
            df['LIRED'] = LIRED
            df['PINK'] = PINK
            df['RED'] = RED
            df['YELLOW'] = YELLOW
            df['GREEN'] = GREEN
            df['LIGHTBLUE'] = LIGHTBLUE
            df['笑脸信号_做多'] = 笑脸信号_做多
            df['笑脸信号_做空'] = 笑脸信号_做空
            
            self.df = df
            return df
            
        except Exception as e:
            print(f"计算幻影主力指标时出错: {str(e)}")
            return self.df

    def calculate_heatmap_volume(self):
        """
        计算成交量热力图相关指标，包括成交量倍数、黄金柱和倍量柱
        
        返回:
        添加了成交量热力指标的DataFrame
        """
        try:
            df = self.df.copy()
            
            # 计算量比相关指标
            VOLUME = np.array(df['Volume'])
            MA_VOLUME = MA(VOLUME, 5)  # 5日均量
            
            # 计算成交量倍数
            VOL_TIMES = np.zeros_like(VOLUME)
            for i in range(5, len(VOLUME)):
                if MA_VOLUME[i] > 0:
                    VOL_TIMES[i] = VOLUME[i] / MA_VOLUME[i]
            
            # 计算黄金柱：当日成交量是5日均量的2倍以上
            GOLD_VOL = (VOL_TIMES >= 2.0)
            
            # 计算倍量柱：当日成交量是前一日的2倍以上
            DOUBLE_VOL = np.zeros_like(VOLUME, dtype=bool)
            for i in range(1, len(VOLUME)):
                if VOLUME[i-1] > 0 and VOLUME[i] >= 2 * VOLUME[i-1]:
                    DOUBLE_VOL[i] = True
            
            # 计算成交量热力值
            HVOL_COLOR = np.zeros_like(VOLUME)
            for i in range(5, len(VOLUME)):
                if VOL_TIMES[i] >= 2.0:  # 黄金柱
                    HVOL_COLOR[i] = 100
                elif VOL_TIMES[i] >= 1.5:  # 放量
                    HVOL_COLOR[i] = 80
                elif VOL_TIMES[i] >= 1.2:  # 小放量
                    HVOL_COLOR[i] = 60
                elif VOL_TIMES[i] <= 0.5:  # 缩量
                    HVOL_COLOR[i] = 20
                elif VOL_TIMES[i] <= 0.8:  # 小缩量
                    HVOL_COLOR[i] = 40
                else:  # 平量
                    HVOL_COLOR[i] = 50
            
            # 将结果添加到DataFrame
            df['VOL_TIMES'] = VOL_TIMES
            df['GOLD_VOL'] = GOLD_VOL
            df['DOUBLE_VOL'] = DOUBLE_VOL
            df['HVOL_COLOR'] = HVOL_COLOR
            
            self.df = df
            return df
            
        except Exception as e:
            print(f"计算热力图成交量指标时出错: {str(e)}")
            return self.df

    def calculate_macd_signals(self):
        """
        计算MACD相关信号，包括零轴上下金叉死叉、先机信号以及底背离顶背离
        
        返回:
        添加了MACD相关信号的DataFrame
        """
        try:
            df = self.df.copy()
            
            # 计算MACD基础指标
            CLOSE = np.array(df['Close'])
            # 使用MyTT库的MACD函数
            DIF, DEA, MACD_VALUE = MACD(CLOSE, SHORT=12, LONG=26, M=9)
            EMAMACD = EMA(MACD_VALUE, 9)
            
            # 检测MACD信号
            零轴下金叉 = np.zeros_like(DIF, dtype=int)
            零轴上金叉 = np.zeros_like(DIF, dtype=int)
            零轴上死叉 = np.zeros_like(DIF, dtype=int)
            零轴下死叉 = np.zeros_like(DIF, dtype=int)
            先机信号 = np.zeros_like(DIF, dtype=int)
            底背离 = np.zeros_like(DIF, dtype=int)
            顶背离 = np.zeros_like(DIF, dtype=int)
            
            for i in range(1, len(DIF)):
                # 零轴下金叉：DIF、DEA都在0轴下，DIF上穿DEA
                if DIF[i] < 0 and DEA[i] < 0 and DIF[i] > DEA[i] and DIF[i-1] <= DEA[i-1]:
                    零轴下金叉[i] = 1
                
                # 零轴上金叉：DIF、DEA都在0轴上，DIF上穿DEA
                if DIF[i] > 0 and DEA[i] > 0 and DIF[i] > DEA[i] and DIF[i-1] <= DEA[i-1]:
                    零轴上金叉[i] = 1
                
                # 零轴上死叉：DIF、DEA都在0轴上，DIF下穿DEA
                if DIF[i] > 0 and DEA[i] > 0 and DIF[i] < DEA[i] and DIF[i-1] >= DEA[i-1]:
                    零轴上死叉[i] = 1
                
                # 零轴下死叉：DIF、DEA都在0轴下，DIF下穿DEA
                if DIF[i] < 0 and DEA[i] < 0 and DIF[i] < DEA[i] and DIF[i-1] >= DEA[i-1]:
                    零轴下死叉[i] = 1
            
            # 先机信号：MACD由负转正
            for i in range(2, len(MACD_VALUE)):
                if MACD_VALUE[i] > 0 and MACD_VALUE[i-1] <= 0:
                    先机信号[i] = 1
            
            # 背离识别 - 这是一个简化版，实际交易中可能需要更复杂的逻辑
            for i in range(20, len(DIF)):
                # 寻找近期的一个高点和一个低点
                high_idx = np.argmax(CLOSE[i-20:i])
                low_idx = np.argmin(CLOSE[i-20:i])
                
                # 底背离：价格创新低但MACD没有创新低
                if low_idx == 19:  # 如果最近的点是最低点
                    if MACD_VALUE[i] > MACD_VALUE[i-10]:  # 简化的MACD比较
                        底背离[i] = 1
                
                # 顶背离：价格创新高但MACD没有创新高
                if high_idx == 19:  # 如果最近的点是最高点
                    if MACD_VALUE[i] < MACD_VALUE[i-10]:  # 简化的MACD比较
                        顶背离[i] = 1
            
            # 将结果添加到DataFrame
            df['DIF'] = DIF
            df['DEA'] = DEA
            df['MACD'] = MACD_VALUE
            df['EMAMACD'] = EMAMACD
            df['零轴下金叉'] = 零轴下金叉
            df['零轴上金叉'] = 零轴上金叉
            df['零轴上死叉'] = 零轴上死叉
            df['零轴下死叉'] = 零轴下死叉
            df['先机信号'] = 先机信号
            df['底背离'] = 底背离
            df['顶背离'] = 顶背离
            
            self.df = df
            return df
            
        except Exception as e:
            print(f"计算MACD相关信号时出错: {str(e)}")
            # 注意：这里返回的是原始的dataframe，而不是修改过的df
            return self.df

# 使用示例
if __name__ == "__main__":
    # 假设我们有一个股票数据DataFrame
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # 创建示例数据
    dates = [datetime.now() - timedelta(days=i) for i in range(100, 0, -1)]
    data = {
        'Date': dates,
        'Open': np.random.normal(10, 1, 100),
        'High': np.random.normal(11, 1, 100),
        'Low': np.random.normal(9, 1, 100),
        'Close': np.random.normal(10, 1, 100),
        'Volume': np.random.normal(1000000, 200000, 100)
    }
    
    # 确保High总是最高的，Low总是最低的
    for i in range(len(data['High'])):
        values = [data['Open'][i], data['Close'][i], data['High'][i], data['Low'][i]]
        data['High'][i] = max(values)
        data['Low'][i] = min(values)
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    
    # 分析股票
    analyzer = StockAnalysis(df)
    result = analyzer.calculate_phantom_indicators()
    result = analyzer.calculate_heatmap_volume()
    result = analyzer.calculate_macd_signals()
    
    # 显示结果
    print("\n计算结果的前几行:")
    print(result[['Close', 'PINK', 'BLUE', 'GOLD_VOL', 'DIF', 'DEA', 'MACD']].tail())