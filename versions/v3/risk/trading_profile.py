#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统一交易风控参数配置（本地持久化）"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any


DEFAULT_TRADING_PROFILE: Dict[str, Any] = {
    "atr_stop_multiplier": 2.0,
    "max_stop_loss_pct": 8.0,
    "target_cap_pct": 15.0,
    "strong_signal_target_boost": 1.2,
    "rr_high": 2.0,
    "rr_mid": 1.5,
    "prob_high": 0.55,
    "prob_mid": 0.52,
    "position_high_pct": 15.0,
    "position_mid_pct": 10.0,
    "position_low_pct": 5.0,
}


def _profile_path() -> Path:
    return Path(__file__).resolve().parent.parent / "db" / "trading_profile.json"


def _sanitize(profile: Dict[str, Any]) -> Dict[str, Any]:
    data = DEFAULT_TRADING_PROFILE.copy()
    if isinstance(profile, dict):
        for k, v in profile.items():
            if k in data:
                try:
                    data[k] = float(v)
                except Exception:
                    pass
    return data


def load_trading_profile() -> Dict[str, Any]:
    path = _profile_path()
    if not path.exists():
        return DEFAULT_TRADING_PROFILE.copy()
    try:
        with open(path, "r") as f:
            obj = json.load(f)
        return _sanitize(obj)
    except Exception:
        return DEFAULT_TRADING_PROFILE.copy()


def save_trading_profile(profile: Dict[str, Any]) -> bool:
    path = _profile_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _sanitize(profile)
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception:
        return False

