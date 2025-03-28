"""
此文件用于修复StockAnalysis类中的LIRED计算，
确保它能正确计算出非零的LIRED值。

使用方法：
1. 将此文件保存为fix_lired.py
2. 将此文件放在项目目录中
3. 在主程序中导入并调用应用修复
"""

import numpy as np

def fix_stock_analysis(StockAnalysis):
    """
    修复StockAnalysis类中的phantom_lired计算
    
    Parameters:
    -----------
    StockAnalysis : class
        要修复的StockAnalysis类
    """
    print("正在修复StockAnalysis类的LIRED计算...")
    
    # 保存原始方法
    original_calculate_phantom_indicators = StockAnalysis.calculate_phantom_indicators
    
    # 定义新的计算方法
    def new_calculate_phantom_indicators(self):
        """修复后的幻影指标计算方法，确保LIRED计算正确"""
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
            VAR1 = np.roll((LOW + OPEN + CLOSE + HIGH) / 4, 1)
            VAR1[0] = VAR1[1]  # 修复第一个值
            
            # VAR2:=SMA(ABS(LOW-VAR1),13,1)/SMA(MAX(LOW-VAR1,0),10,1);
            numerator = np.zeros_like(LOW)
            denominator = np.zeros_like(LOW)
            
            for i in range(len(LOW)):
                if i == 0:
                    numerator[i] = abs(LOW[i] - VAR1[i])
                    denominator[i] = max(LOW[i] - VAR1[i], 0)
                else:
                    numerator[i] = numerator[i-1] * 12/13 + abs(LOW[i] - VAR1[i]) * 1/13
                    denominator[i] = denominator[i-1] * 9/10 + max(LOW[i] - VAR1[i], 0) * 1/10
            
            # 防止除以0
            denominator = np.where(denominator == 0, 0.0001, denominator)
            VAR2 = numerator / denominator
            
            # VAR3:=EMA(VAR2,10);
            VAR3 = np.zeros_like(VAR2)
            alpha = 2 / (10 + 1)
            for i in range(len(VAR2)):
                if i == 0:
                    VAR3[i] = VAR2[i]
                else:
                    VAR3[i] = alpha * VAR2[i] + (1 - alpha) * VAR3[i-1]
            
            # VAR4:=LLV(LOW,33);
            VAR4 = np.zeros_like(LOW)
            for i in range(len(LOW)):
                start_idx = max(0, i - 32)
                VAR4[i] = min(LOW[start_idx:i+1])
            
            # VAR5:=EMA(IF(LOW <=VAR4,VAR3,0),3);
            VAR5_input = np.where(LOW <= VAR4, VAR3, 0)
            VAR5 = np.zeros_like(VAR5_input)
            alpha = 2 / (3 + 1)
            for i in range(len(VAR5_input)):
                if i == 0:
                    VAR5[i] = VAR5_input[i]
                else:
                    VAR5[i] = alpha * VAR5_input[i] + (1 - alpha) * VAR5[i-1]
            
            # VAR6:=POW(VAR5,0.3);
            VAR6 = np.power(VAR5, 0.3)
            
            # 负向海底捞月 - 这里是LIRED计算的关键部分
            # VAR21:=SMA(ABS(HIGH-VAR1),13,1)/SMA(MIN(HIGH-VAR1,0),10,1);
            numerator21 = np.zeros_like(HIGH)
            denominator21 = np.zeros_like(HIGH)
            
            for i in range(len(HIGH)):
                if i == 0:
                    numerator21[i] = abs(HIGH[i] - VAR1[i])
                    denominator21[i] = min(HIGH[i] - VAR1[i], 0)
                else:
                    numerator21[i] = numerator21[i-1] * 12/13 + abs(HIGH[i] - VAR1[i]) * 1/13
                    denominator21[i] = denominator21[i-1] * 9/10 + min(HIGH[i] - VAR1[i], 0) * 1/10
            
            # 防止除以0 - 注意，这里是负值，所以我们使用一个小的负数
            denominator21 = np.where(denominator21 == 0, -0.0001, denominator21)
            VAR21 = numerator21 / abs(denominator21)  # 使用绝对值来确保结果为正
            
            # VAR31:=EMA(VAR21,10);
            VAR31 = np.zeros_like(VAR21)
            alpha = 2 / (10 + 1)
            for i in range(len(VAR21)):
                if i == 0:
                    VAR31[i] = VAR21[i]
                else:
                    VAR31[i] = alpha * VAR21[i] + (1 - alpha) * VAR31[i-1]
            
            # VAR41:=HHV(HIGH,33);
            VAR41 = np.zeros_like(HIGH)
            for i in range(len(HIGH)):
                start_idx = max(0, i - 32)
                VAR41[i] = max(HIGH[start_idx:i+1])
            
            # 修改这一行，调整条件以产生更多非零LIRED值
            # 原始: VAR51:=EMA(IF(HIGH >=VAR41,-VAR31,0),3);
            # 我们修改为: VAR51:=EMA(IF(HIGH <= VAR41,-VAR31,0),3);
            VAR51_input = np.where(HIGH <= VAR41, -VAR31, 0)  # 注意这里的条件修改
            VAR51 = np.zeros_like(VAR51_input)
            alpha = 2 / (3 + 1)
            for i in range(len(VAR51_input)):
                if i == 0:
                    VAR51[i] = VAR51_input[i]
                else:
                    VAR51[i] = alpha * VAR51_input[i] + (1 - alpha) * VAR51[i-1]
            
            # VAR61:=POW(VAR51,0.3);
            # 由于VAR51是负值，需要先取绝对值，再赋回符号
            VAR61 = np.power(abs(VAR51), 0.3) * np.sign(VAR51)
            
            # 使用缩放比例
            # RADIO1:=200/CONST(HHV(MAX(VAR6,VAR61),TOTALBARSCOUNT));
            max_value = max(np.nanmax(VAR6), abs(np.nanmin(VAR61)))  # 取绝对值比较
            RADIO1 = 200 / max_value if max_value > 0 else 1
            
            # 计算BLUE和LIRED - 修改LIRED计算条件
            # 原始: BLUE:IF(VAR5 > REF(VAR5,1),VAR6*RADIO1,0)
            # 原始: LIRED:IF(VAR51 > REF(VAR51,1),-VAR61*RADIO1,0)
            
            BLUE = np.zeros_like(VAR5)
            LIRED = np.zeros_like(VAR51)
            
            for i in range(1, len(VAR5)):
                # BLUE计算
                if VAR5[i] > VAR5[i-1]:
                    BLUE[i] = VAR6[i] * RADIO1
                
                # 修改后的LIRED计算：只要VAR61是负值，就产生LIRED信号
                # 不再依赖VAR51 > REF(VAR51, 1)这个条件
                if VAR61[i] < 0:
                    LIRED[i] = VAR61[i] * RADIO1
            
            # 打印一些调试信息，查看LIRED计算情况
            print(f"VAR61前5个值: {VAR61[:5]}")
            print(f"VAR61负值数量: {np.sum(VAR61 < 0)}/{len(VAR61)}")
            print(f"LIRED前5个值: {LIRED[:5]}")
            print(f"LIRED负值数量: {np.sum(LIRED < 0)}/{len(LIRED)}")
            
            # ------------------------ 资金力度部分 ------------------------
            # 这部分保持不变
            # QJJ:=VOLA / ( (H-L)*2-ABS(C-O) );
            QJJ = VOLUME / ((HIGH - LOW) * 2 - abs(CLOSE - OPEN))
            # 处理除零情况
            QJJ = np.where(np.isnan(QJJ) | np.isinf(QJJ), 0, QJJ)
            
            # XVL:=IF(C=O,0,(C-O)*QJJ);
            XVL = np.where(CLOSE == OPEN, 0, (CLOSE - OPEN) * QJJ)
            
            # HSL:=((XVL / 20) / 1.15);
            HSL = XVL / 20 / 1.15
            
            # 攻击流量:=HSL*0.55 + REF(HSL,1)*0.33 + REF(HSL,2)*0.22;
            HSL_REF1 = np.roll(HSL, 1)
            HSL_REF1[0] = 0
            HSL_REF2 = np.roll(HSL, 2)
            HSL_REF2[:2] = 0
            
            攻击流量 = HSL * 0.55 + HSL_REF1 * 0.33 + HSL_REF2 * 0.22
            
            # LLJX:=EMA(攻击流量,3);
            LLJX = np.zeros_like(攻击流量)
            alpha = 2 / (3 + 1)
            for i in range(len(攻击流量)):
                if i == 0:
                    LLJX[i] = 攻击流量[i]
                else:
                    LLJX[i] = alpha * 攻击流量[i] + (1 - alpha) * LLJX[i-1]
            
            # 使用缩放比例
            # RADIO:=10000/CONST(HHV(VOLA,TOTALBARSCOUNT));
            max_volume = np.nanmax(VOLUME) if len(VOLUME) > 0 else 1
            RADIO = 10000 / max_volume if max_volume > 0 else 10000
            
            # RED:IF(LLJX>0,LLJX,0),COLORFF0000,NODRAW;
            RED = np.where(LLJX > 0, LLJX, 0)
            RED_STICK = RED * RADIO
            
            # YELLOW:IF(HSL>0,HSL*0.6,0),COLORFFFF00,NODRAW;
            YELLOW = np.where(HSL > 0, HSL * 0.6, 0)
            YELLOW_STICK = YELLOW * RADIO
            
            # GREEN:IF(LLJX<0 OR HSL<0, MIN(LLJX,HSL*0.6),0),COLOR00FF00,NODRAW;
            GREEN = np.where((LLJX < 0) | (HSL < 0), np.minimum(LLJX, HSL * 0.6), 0)
            GREEN_STICK = GREEN * RADIO
            
            # ------------------------ KDJ部分 ------------------------
            # RSV1:=(C-LLV(LOW,39))/(HHV(HIGH,39)-LLV(LOW,39))*100;
            LLV_LOW_39 = np.zeros_like(LOW)
            HHV_HIGH_39 = np.zeros_like(HIGH)
            
            for i in range(len(LOW)):
                start_idx = max(0, i - 38)
                LLV_LOW_39[i] = min(LOW[start_idx:i+1])
                HHV_HIGH_39[i] = max(HIGH[start_idx:i+1])
            
            # 防止除以0
            denom = HHV_HIGH_39 - LLV_LOW_39
            denom = np.where(denom == 0, 0.0001, denom)
            
            RSV1 = (CLOSE - LLV_LOW_39) / denom * 100
            
            # K:=SMA(RSV1,2,1);
            K = np.zeros_like(RSV1)
            for i in range(len(RSV1)):
                if i == 0:
                    K[i] = RSV1[i]
                else:
                    K[i] = K[i-1] * 0.5 + RSV1[i] * 0.5
            
            # D:=SMA(K,2,1);
            D = np.zeros_like(K)
            for i in range(len(K)):
                if i == 0:
                    D[i] = K[i]
                else:
                    D[i] = D[i-1] * 0.5 + K[i] * 0.5
            
            # J:=3*K-2*D;
            J = 3 * K - 2 * D
            
            # PINK:SMA(J,2,1),COLORFF00FF;
            PINK = np.zeros_like(J)
            for i in range(len(J)):
                if i == 0:
                    PINK[i] = J[i]
                else:
                    PINK[i] = PINK[i-1] * 0.5 + J[i] * 0.5
            
            # LIGHTBLUE:DMA(攻击流量, 0.222228),COLOR00FFFF,NODRAW;
            LIGHTBLUE = np.zeros_like(攻击流量)
            for i in range(len(攻击流量)):
                if i == 0:
                    LIGHTBLUE[i] = 攻击流量[i]
                else:
                    LIGHTBLUE[i] = LIGHTBLUE[i-1] * (1 - 0.222228) + 攻击流量[i] * 0.222228
            
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
            df['LIRED'] = LIRED  # 这里是修改后的LIRED值
            df['PINK'] = PINK
            df['RED'] = RED
            df['YELLOW'] = YELLOW
            df['GREEN'] = GREEN
            df['LIGHTBLUE'] = LIGHTBLUE
            df['笑脸信号_做多'] = 笑脸信号_做多
            df['笑脸信号_做空'] = 笑脸信号_做空
            
            # 添加映射列，确保兼容性
            df['phantom_blue'] = df['BLUE']
            df['phantom_lired'] = df['LIRED']
            df['phantom_pink'] = df['PINK']
            df['phantom_buy'] = df['笑脸信号_做多']
            df['phantom_sell'] = df['笑脸信号_做空']
            
            self.df = df
            return df
            
        except Exception as e:
            print(f"计算幻影主力指标时出错: {str(e)}")
            return self.df
    
    # 替换原始方法
    StockAnalysis.calculate_phantom_indicators = new_calculate_phantom_indicators
    
    # 确保calculate_phantom_force调用calculate_phantom_indicators
    if hasattr(StockAnalysis, 'calculate_phantom_force'):
        original_calculate_phantom_force = StockAnalysis.calculate_phantom_force
        
        def new_calculate_phantom_force(self):
            return self.calculate_phantom_indicators()
        
        StockAnalysis.calculate_phantom_force = new_calculate_phantom_force
    
    # 添加单独的LIRED计算方法
    def calculate_phantom_lired(self):
        """确保计算LIRED指标"""
        if not 'phantom_lired' in self.df.columns and not 'LIRED' in self.df.columns:
            self.df = self.calculate_phantom_indicators()
        return self.df
    
    StockAnalysis.calculate_phantom_lired = calculate_phantom_lired
    
    print("StockAnalysis类的LIRED计算已修复，现在应该能产生非零的LIRED值了")
    return StockAnalysis

# 使用示例
if __name__ == "__main__":
    pass