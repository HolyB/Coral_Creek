---
description: 每日买卖点信号系统使用指南
---

## 信号系统架构

### 多头信号（买入）
| 信号 | 来源 | 字段 | 阈值 |
|---|---|---|---|
| BLUE 海底捞月 | `indicator_utils.calculate_blue_signal_series()` | `blue_daily/weekly/monthly` | >100 强信号 |
| BLUE 消失买点 | `indicator_utils.calculate_phantom_indicator()` | `blue_disappear` | True |
| 黑马信号 | `indicator_utils.calculate_heima_signal_series()` | `heima_daily/weekly/monthly` | True |
| 掘地信号 | 同上 | `juedi_daily/weekly/monthly` | True |
| 黄金底 | CCI 极度超卖 + 底部金叉 | 动态计算 | CCI < -100 |
| 多空王买入 | KDJ+RSI+九转 | `duokongwang_buy` | True |
| PINK 进场 | KDJ变体上穿10 | `pink_daily` | < 10 |

### 空头信号（卖出/逃顶）
| 信号 | 来源 | 字段 | 阈值 |
|---|---|---|---|
| PINK 超买 | KDJ变体 | `pink_daily` | > 90 逃顶 |
| LIRED 逃顶 | 负向海底捞月 | `lired_daily` | > 0 |
| 多空王卖出 | KDJ+RSI 过热 | `duokongwang_sell` | True |
| LIRED 消失 | 负向海底捞月消失 | `lired_disappear` | True (卖点) |

### 数据流
```
scan_service.py (每日扫描)
  → indicator_utils.py (计算指标)
  → database.py / supabase_db.py (存储)
  → app.py (展示)
  → candidate_tracking_service.py (追踪历史表现)
```

### 原始通达信公式对照
- BLUE = `IF(VAR5 > REF(VAR5,1), VAR6*RADIO1, 0)` → 正值蓝色柱
- LIRED = `IF(VAR51 > REF(VAR51,1), -VAR61*RADIO1, 0)` → 负值粉红柱
- PINK = `SMA(J,2,1)` → 0-100 范围, >90 逃顶, <10 进场
- 核心代码在 `versions/v3/indicator_utils.py` 的 `calculate_phantom_indicator()`
