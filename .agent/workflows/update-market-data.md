---
description: 更新本地市场历史数据 (A股/美股 1d) - 解压 zip + 导入 stock_history.db
---

# 本地市场数据更新流程

## 数据源

本地市场数据存放在以下位置：

| 市场 | 原始 zip | 解压后 CSV | 
|------|----------|------------|
| A股  | `/Users/bertwang/Cursor/A股/1d/{月份}/YYYYMMDD.zip` | `/Users/bertwang/Cursor/A股/1d_unzip/{月份}/YYYYMMDD.csv` |
| 美股 | `/Users/bertwang/Cursor/美股数据/1d/{月份}/YYYYMMDD.zip` | `/Users/bertwang/Cursor/美股数据/1d_unzip/{月份}/YYYYMMDD.csv` |

> [!IMPORTANT]
> - 月份文件夹格式: `YYYYMM`（如 `202603`）；2016年及以前用年份文件夹 `YYYY`
> - zip 文件使用 AES-256 加密，密码 = `SHA256(文件名 + "vvtr123!@#qwe")`
> - 需要 `pyzipper` 库来解压（`pip install pyzipper`）

## CSV 格式

**A股** (exchange → symbol 后缀: SHSE→.SH, SZSE→.SZ):
```
exchange,symbol,open,close,high,low,amount,volume,bob,eob,type,sequence
SHSE,600000,9.74,9.89,9.9,9.71,714778139.88,72726024.0,...
```

**美股** (注意列顺序不同: open,high,low,close):
```
exchange,symbol,open,high,low,close,amount,volume,bob,eob,type,sequence
XNYS,AA,58.26,60.09,56.75,59.65,329064134.83,5577141.0,...
```

## 操作步骤

### 1. 确认需要更新的月份

查看 stock_history.db 当前最新日期：

// turbo
```bash
cd /Users/bertwang/Cursor/Coral_Creek/versions/v3 && PYTHONPATH=. /Users/bertwang/miniconda3/bin/python3 scripts/import_market_data.py --stats
```

### 2. 确认 zip 文件已下载

检查原始 zip 目录中是否有新文件（替换 `YYYYMM` 为目标月份）：
```bash
ls /Users/bertwang/Cursor/A股/1d/YYYYMM/
ls /Users/bertwang/Cursor/美股数据/1d/YYYYMM/
```

### 3. 解压 + 导入（一键）

// turbo
```bash
cd /Users/bertwang/Cursor/Coral_Creek/versions/v3 && PYTHONPATH=. /Users/bertwang/miniconda3/bin/python3 scripts/import_market_data.py --month YYYYMM --unzip
```

也可以分步操作：

#### 3a. 只解压
```bash
cd /Users/bertwang/Cursor/Coral_Creek/versions/v3 && PYTHONPATH=. /Users/bertwang/miniconda3/bin/python3 scripts/unzip_market_data.py --month YYYYMM
```

#### 3b. 只导入
```bash
cd /Users/bertwang/Cursor/Coral_Creek/versions/v3 && PYTHONPATH=. /Users/bertwang/miniconda3/bin/python3 scripts/import_market_data.py --month YYYYMM
```

#### 3c. 导入单天
```bash
cd /Users/bertwang/Cursor/Coral_Creek/versions/v3 && PYTHONPATH=. /Users/bertwang/miniconda3/bin/python3 scripts/import_market_data.py --date YYYYMMDD
```

#### 3d. 只导入特定市场
```bash
cd /Users/bertwang/Cursor/Coral_Creek/versions/v3 && PYTHONPATH=. /Users/bertwang/miniconda3/bin/python3 scripts/import_market_data.py --month YYYYMM --market CN
```

### 4. 验证导入结果

// turbo
```bash
cd /Users/bertwang/Cursor/Coral_Creek/versions/v3 && PYTHONPATH=. /Users/bertwang/miniconda3/bin/python3 scripts/import_market_data.py --stats
```

## 相关脚本

| 脚本 | 功能 |
|------|------|
| `scripts/unzip_market_data.py` | 解压 AES 加密的 zip 数据文件 |
| `scripts/import_market_data.py` | 解析 CSV 并导入 stock_history.db |
| `ml/fetch_history.py` | 从 API (Polygon/yfinance) 获取历史数据（备用） |
| `db/stock_history.py` | stock_history.db 的 ORM 操作 |

## 注意事项

> [!WARNING]
> - 导入使用 `INSERT OR REPLACE`，重复导入同一天不会出错
> - A股带 (1)/(2) 后缀的是重复文件，脚本会自动去重
> - 美股 symbol 直接用原始 ticker（如 AAPL），A股转为 `000001.SZ` 格式
> - 如果 Polygon/yfinance 也已经写入了某些数据，CSV 导入会覆盖更新
