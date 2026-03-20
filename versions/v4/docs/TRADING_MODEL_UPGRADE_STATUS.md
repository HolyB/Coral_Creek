# Trading Model Upgrade Status (v3)

Last update: 2026-02-07

## Completed

### 1) Objective Unification (Model + Backtest)
- Return predictor horizons unified to `5d/20d/60d` (mid/long focused).
- Training targets for `20d/60d` upgraded to:
  - cross-sectional excess return by scan date
  - minus drawdown penalty
- Backtester net return aligned with round-trip costs:
  - one-side commission + one-side slippage
  - applied on both entry and exit

## 2) Feature Stability + Walk-forward
- Added feature stability report generation in training pipeline:
  - missing rate
  - drift score
  - Spearman IC (20d/60d)
  - stable/unstable feature ranking
- Added walk-forward evaluation:
  - rolling train/test by date groups
  - metrics: avg Spearman IC, direction accuracy, top20 return
- Artifacts produced in `ml/saved_models/v2_<market>/`:
  - `training_objective.json`
  - `feature_stability_report.json`
  - `walk_forward_report.json`

### 3) Strategy Combination Layer
- Master strategy summary supports profile-aware weighting:
  - `short`, `medium`, `long`
- Dynamic weight adjustment by BLUE regime added.
- UI strategy profile selector added in scan/deep-analysis flow.

### 4) Risk Controls Productization
- Added persistent trading risk profile config:
  - file: `risk/trading_profile.py`
- SmartPicker now reads global risk profile for:
  - ATR stop multiplier
  - max stop cap
  - target cap & strong-signal boost
  - RR/probability thresholds
  - position sizing tiers
- Added ML page UI to edit/save risk profile.

## In Progress

### 5) Execution Layer Robustness
- Planned:
  - order simulation timing/market-hours checks consistency
  - stricter fill assumptions for paper/live parity

### 6) Strategy Portfolio Governance
- Planned:
  - strategy-level allocation caps
  - per-theme concentration caps
  - rolling strategy weight decay/reallocation rules

## How To Validate Quickly
1. Re-train once:
   - `python ml/pipeline.py --market US --days 180`
2. Open `ML智能选股` page:
   - check dynamic horizons from model metadata
   - check `🧪 稳定性` tab metrics
   - adjust/save `🛡️ 风控参数`
3. Open scan deep-analysis:
   - switch strategy profile `short/medium/long`
   - compare consensus changes
