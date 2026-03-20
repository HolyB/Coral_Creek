# Components package
from .stock_detail import render_unified_stock_detail

# Alpaca trading widgets
try:
    from .alpaca_widget import (
        render_alpaca_sidebar_widget,
        render_alpaca_floating_bar,
        render_alpaca_quick_trade,
        render_inline_backtest
    )
except ImportError:
    pass  # Will be available once alpaca-py is installed
