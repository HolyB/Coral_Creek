-- Supabase migration for candidate_tracking (align with local SQLite schema)
-- Run in Supabase SQL Editor

ALTER TABLE public.candidate_tracking
  ADD COLUMN IF NOT EXISTS source TEXT DEFAULT 'daily_scan',
  ADD COLUMN IF NOT EXISTS signal_price DOUBLE PRECISION,
  ADD COLUMN IF NOT EXISTS current_price DOUBLE PRECISION,
  ADD COLUMN IF NOT EXISTS pnl_pct DOUBLE PRECISION DEFAULT 0,
  ADD COLUMN IF NOT EXISTS days_since_signal INTEGER DEFAULT 0,
  ADD COLUMN IF NOT EXISTS first_positive_day INTEGER,
  ADD COLUMN IF NOT EXISTS first_nonpositive_after_positive_day INTEGER,
  ADD COLUMN IF NOT EXISTS max_up_pct DOUBLE PRECISION DEFAULT 0,
  ADD COLUMN IF NOT EXISTS max_drawdown_pct DOUBLE PRECISION DEFAULT 0,
  ADD COLUMN IF NOT EXISTS pnl_d1 DOUBLE PRECISION,
  ADD COLUMN IF NOT EXISTS pnl_d3 DOUBLE PRECISION,
  ADD COLUMN IF NOT EXISTS pnl_d5 DOUBLE PRECISION,
  ADD COLUMN IF NOT EXISTS pnl_d10 DOUBLE PRECISION,
  ADD COLUMN IF NOT EXISTS pnl_d20 DOUBLE PRECISION,
  ADD COLUMN IF NOT EXISTS cap_category TEXT,
  ADD COLUMN IF NOT EXISTS industry TEXT,
  ADD COLUMN IF NOT EXISTS blue_daily DOUBLE PRECISION,
  ADD COLUMN IF NOT EXISTS blue_weekly DOUBLE PRECISION,
  ADD COLUMN IF NOT EXISTS blue_monthly DOUBLE PRECISION,
  ADD COLUMN IF NOT EXISTS heima_daily BOOLEAN,
  ADD COLUMN IF NOT EXISTS heima_weekly BOOLEAN,
  ADD COLUMN IF NOT EXISTS heima_monthly BOOLEAN,
  ADD COLUMN IF NOT EXISTS juedi_daily BOOLEAN,
  ADD COLUMN IF NOT EXISTS juedi_weekly BOOLEAN,
  ADD COLUMN IF NOT EXISTS juedi_monthly BOOLEAN,
  ADD COLUMN IF NOT EXISTS vp_rating TEXT,
  ADD COLUMN IF NOT EXISTS profit_ratio DOUBLE PRECISION,
  ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'tracking';

CREATE UNIQUE INDEX IF NOT EXISTS uq_candidate_tracking_symbol_market_date
ON public.candidate_tracking(symbol, market, signal_date);

CREATE INDEX IF NOT EXISTS idx_candidate_tracking_date
ON public.candidate_tracking(signal_date);

CREATE INDEX IF NOT EXISTS idx_candidate_tracking_market
ON public.candidate_tracking(market);

CREATE INDEX IF NOT EXISTS idx_candidate_tracking_status
ON public.candidate_tracking(status);
