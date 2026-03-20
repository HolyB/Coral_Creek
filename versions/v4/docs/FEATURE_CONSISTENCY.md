# Coral Creek V3 - Feature & Indicator 一致性报告

## 📊 统一来源 (Single Source of Truth)

所有技术指标应从 `indicator_utils.py` 导入使用。

---

## ✅ 已统一的指标

### 1. BLUE 信号 (海底捞月)
| 函数 | 位置 | 状态 |
|------|------|------|
| `calculate_blue_signal_series()` | `indicator_utils.py:184` | ✅ **主版本** |
| `calculate_blue_signal()` | `chart_utils.py:13` | ✅ 已改为调用主版本 |
| `calculate_blue_signal()` | `scripts/scan_blue_baseline_v2.py:71` | ✅ 已改为调用主版本 |

**算法说明:**
```
BLUE = IF(VAR5 > REF(VAR5,1), VAR6 * RADIO1, 0)
RADIO1 = 200 / max(VAR6, |VAR61|)  # 考虑多空能量平衡
```

### 2. 黑马信号 (HEIMA)
| 函数 | 位置 | 状态 |
|------|------|------|
| `calculate_heima_signal_series()` | `indicator_utils.py:250` | ✅ 主版本 |
| `calculate_heima_full()` | `indicator_utils.py:371` | ✅ 完整版 |

### 3. KDJ 指标
| 函数 | 位置 | 状态 |
|------|------|------|
| `calculate_kdj_series()` | `indicator_utils.py:489` | ✅ 主版本 |

### 4. ATR 指标
| 函数 | 位置 | 状态 |
|------|------|------|
| `calculate_atr_series()` | `indicator_utils.py:519` | ✅ 主版本 |

### 5. ADX 趋势强度
| 函数 | 位置 | 状态 |
|------|------|------|
| `calculate_adx_series()` | `indicator_utils.py:538` | ✅ 主版本 |

### 6. 筹码分布
| 函数 | 位置 | 状态 |
|------|------|------|
| `calculate_volume_profile_metrics()` | `indicator_utils.py:588` | ✅ 主版本 |

### 7. 波动率
| 函数 | 位置 | 状态 |
|------|------|------|
| `calculate_volatility()` | `indicator_utils.py:686` | ✅ 主版本 |

### 8. ZigZag
| 函数 | 位置 | 状态 |
|------|------|------|
| `calculate_zigzag()` | `indicator_utils.py:703` | ✅ 主版本 |

### 9. 波浪分析
| 函数 | 位置 | 状态 |
|------|------|------|
| `analyze_elliott_wave_proxy()` | `indicator_utils.py:792` | ✅ 主版本 |

### 10. 缠论分析
| 函数 | 位置 | 状态 |
|------|------|------|
| `analyze_chanlun_proxy()` | `indicator_utils.py:885` | ✅ 主版本 |

### 11. 幻影主力
| 函数 | 位置 | 状态 |
|------|------|------|
| `calculate_phantom_indicator()` | `indicator_utils.py:71` | ✅ 主版本 |

---

## 📦 数据库字段

### scan_results 表
| 字段 | 说明 | 计算来源 |
|------|------|----------|
| `blue_daily` | 日线 BLUE 值 | `calculate_blue_signal_series()` |
| `blue_weekly` | 周线 BLUE 值 | `calculate_blue_signal_series()` |
| `blue_monthly` | 月线 BLUE 值 | `calculate_blue_signal_series()` |
| `blue_days` | 满足日线条件的天数 | 统计 BLUE > 100 的天数 |
| `blue_weeks` | 满足周线条件的周数 | 统计 BLUE > 130 的周数 |
| `is_heima` | 是否有黑马信号 | `calculate_heima_signal_series()` |
| `is_juedi` | 是否有绝地信号 | `calculate_heima_signal_series()` |
| `wave_phase` | 波浪阶段 | `analyze_elliott_wave_proxy()` |
| `chan_signal` | 缠论信号 | `analyze_chanlun_proxy()` |

---

## 🔧 基础工具函数 (indicator_utils.py)

| 函数 | 说明 |
|------|------|
| `REF(series, n)` | 前 n 期值 |
| `EMA(series, n)` | 指数移动平均 |
| `SMA(series, n, m)` | 通达信加权平均 |
| `IF(cond, a, b)` | 条件表达式 |
| `LLV(series, n)` | n 期最低值 |
| `HHV(series, n)` | n 期最高值 |
| `MA(series, n)` | 简单移动平均 |
| `AVEDEV(series, n)` | 平均绝对偏差 |
| `DMA(series, alpha)` | 动态移动平均 |
| `CROSS(a, b)` | a 上穿 b |

---

## 📍 使用示例

```python
from indicator_utils import (
    calculate_blue_signal_series,
    calculate_heima_signal_series,
    calculate_kdj_series,
    calculate_atr_series,
    calculate_adx_series
)

# 计算 BLUE
blue = calculate_blue_signal_series(opens, highs, lows, closes)

# 计算黑马
heima, juedi = calculate_heima_signal_series(highs, lows, closes, opens)

# 计算 KDJ
k, d, j = calculate_kdj_series(highs, lows, closes)
```

---

## ⚠️ 注意事项

1. **不要**在其他文件中重新实现这些指标
2. **总是**从 `indicator_utils` 导入
3. 如需修改算法，只修改 `indicator_utils.py` 主版本
4. BLUE 阈值标准：日线 >= 100，周线 >= 130

---

*Last Updated: 2026-02-06*
