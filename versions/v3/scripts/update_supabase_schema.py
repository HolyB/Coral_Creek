#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""通过 Supabase REST API (rpc) 添加缺失列"""
import requests
import json

SUPABASE_URL = "https://worqpdsypymnzqjbidyz.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndvcnFwZHN5cHltbnpxamJpZHl6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3Njk4MTA5MjksImV4cCI6MjA4NTM4NjkyOX0.UzE54Q4QB1mQZqRp_jn4BWGFOtWN3GAscrmGpHpMG9U"

headers = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
}

# 方法1: 尝试逐个列添加 - 用 insert 测试哪些列存在
# 先获取当前列
r = requests.get(
    f"{SUPABASE_URL}/rest/v1/scan_results?select=*&limit=1",
    headers=headers,
)
print(f"Status: {r.status_code}")
if r.status_code == 200 and r.json():
    existing_cols = set(r.json()[0].keys())
    print(f"Existing columns ({len(existing_cols)}): {sorted(existing_cols)}")
else:
    print(f"Response: {r.text[:200]}")
    existing_cols = set()

# 需要添加的列
needed_cols = {
    "adaptive_thresh": "float8",
    "chan_desc": "text",
    "chan_signal": "text",
    "day_close": "float8",
    "day_high": "float8",
    "day_low": "float8",
    "duokongwang_buy": "bool",
    "duokongwang_sell": "bool",
    "legacy_signal": "bool",
    "profit_ratio": "float8",
    "regime": "text",
    "risk_reward_score": "float8",
    "shares_rec": "int4",
    "stop_loss": "float8",
    "strat_c_resonance": "bool",
    "strat_d_trend": "bool",
    "updated_at": "timestamptz",
    "vp_rating": "text",
    "wave_desc": "text",
    "wave_phase": "text",
}

missing = {k: v for k, v in needed_cols.items() if k not in existing_cols}
print(f"\nMissing columns: {len(missing)}")
for col, typ in sorted(missing.items()):
    print(f"  {col}: {typ}")

if missing:
    print("\n⚠️ 这些列需要手动在 Supabase SQL Editor 中添加")
    print("请登录 https://supabase.com/dashboard/project/worqpdsypymnzqjbidyz/sql/new")
    print("然后运行以下 SQL:\n")
    
    parts = []
    for col, typ in sorted(missing.items()):
        default = " DEFAULT NOW()" if typ == "timestamptz" else ""
        parts.append(f"  ADD COLUMN IF NOT EXISTS {col} {typ}{default}")
    
    sql = "ALTER TABLE scan_results\n" + ",\n".join(parts) + ";"
    print(sql)
    
    print("\n\n--- 或者提供你的 Supabase 数据库密码，我可以直接通过 psycopg2 连接执行 ---")
else:
    print("\n✅ 所有列都已存在！")
