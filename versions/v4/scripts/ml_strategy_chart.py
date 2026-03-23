#!/usr/bin/env python3
"""Generate strategy comparison charts for email reports - v2
Fixes: CJK font, cumulative compounding, multi-horizon bars, English+CN labels
"""
import sqlite3, base64, io
import numpy as np
from pathlib import Path
from collections import defaultdict

V3 = Path(__file__).resolve().parent.parent


def _setup_cjk_font():
    """Try to load a CJK-capable font for matplotlib"""
    import matplotlib.font_manager as fm
    import matplotlib
    # macOS fonts that support CJK
    cjk_candidates = [
        '/System/Library/Fonts/Hiragino Sans GB.ttc',
        '/System/Library/Fonts/PingFang.ttc',
        '/System/Library/Fonts/STHeiti Medium.ttc',
        '/Library/Fonts/Arial Unicode.ttf',
    ]
    for fpath in cjk_candidates:
        if Path(fpath).exists():
            fm.fontManager.addfont(fpath)
            prop = fm.FontProperties(fname=fpath)
            matplotlib.rcParams['font.family'] = prop.get_name()
            return True
    # Fallback: use English only
    return False


def generate_strategy_chart(market, days=60):
    """Generate strategy comparison chart as base64 PNG
    Top: cumulative compounded return per tier
    Bottom: 5D/10D/20D avg return + win rate bars per tier
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    has_cjk = _setup_cjk_font()

    db_path = V3 / 'db' / 'ml_daily_picks.db'
    if not db_path.exists():
        return None

    conn = sqlite3.connect(str(db_path))
    from datetime import datetime, timedelta
    cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    rows = conn.execute(
        """SELECT date, tier, symbol, price, actual_5d, actual_10d, actual_20d
           FROM mmoe_daily_picks WHERE market=? AND date>=?
           ORDER BY date""",
        (market, cutoff)
    ).fetchall()
    conn.close()

    if len(rows) < 3:
        return None

    # Group by date
    date_picks = defaultdict(list)
    for r in rows:
        date_picks[r[0]].append(r)
    dates_sorted = sorted(date_picks.keys())

    # Define strategies with bilingual labels
    if market == 'US':
        strategies = {
            'All Tiers': lambda picks: picks,
            'Mega+Large': lambda picks: [p for p in picks if 'Mega' in p[1] or 'Large' in p[1]],
            'Mid': lambda picks: [p for p in picks if 'Mid' in p[1]],
            'Small+Micro': lambda picks: [p for p in picks if 'Small' in p[1] or 'Micro' in p[1]],
        }
    else:
        strategies = {
            'All Tiers': lambda picks: picks,
            'Large': lambda picks: [p for p in picks if '大盘' in p[1]],
            'Mid': lambda picks: [p for p in picks if '中盘' in p[1]],
            'Small': lambda picks: [p for p in picks if '小盘' in p[1]],
        }

    colors = ['#818cf8', '#f59e0b', '#22c55e', '#ec4899']

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), facecolor='#0f172a',
                              gridspec_kw={'height_ratios': [2.5, 1]})
    ax1, ax2 = axes

    for ax in [ax1, ax2]:
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='#94a3b8', labelsize=8)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_color('#334155')
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

    # ===== Top chart: cumulative additive return =====
    # Each pick's return is added (not compounded) since picks overlap in time
    for (name, filter_fn), color in zip(strategies.items(), colors):
        cum_ret = [0]
        for d in dates_sorted:
            picks = filter_fn(date_picks[d])
            if picks:
                # Use 10D return as primary, fallback 20d/5d
                rets = []
                for p in picks:
                    r = p[5] if p[5] is not None else p[6] if p[6] is not None else p[4]
                    if r is not None:
                        rets.append(r)
                if rets:
                    cum_ret.append(cum_ret[-1] + np.mean(rets))
                else:
                    cum_ret.append(cum_ret[-1])
            else:
                cum_ret.append(cum_ret[-1])

        x = range(len(cum_ret))
        ax1.plot(x, cum_ret, color=color, linewidth=2.2, label=name, alpha=0.9)
        ax1.fill_between(x, 0, cum_ret, color=color, alpha=0.06)

    ax1.axhline(y=0, color='#475569', linewidth=0.5, linestyle='--')
    period_label = f'YTD' if days >= 300 else f'{days}D'
    title = f'{market} Cumulative Return ({period_label}, {len(dates_sorted)} picks)'
    ax1.set_title(title, color='#e2e8f0', fontsize=13, fontweight='bold', pad=10)
    ax1.set_ylabel('Return %', color='#94a3b8', fontsize=9)
    ax1.legend(loc='upper left', fontsize=9, facecolor='#0f172a',
               edgecolor='#334155', labelcolor='#e2e8f0')

    # Add x-axis date labels (first, mid, last)
    if len(dates_sorted) > 2:
        tick_pos = [0, len(dates_sorted) // 2, len(dates_sorted)]
        tick_labels = [dates_sorted[0][5:], dates_sorted[len(dates_sorted)//2][5:], dates_sorted[-1][5:]]
        ax1.set_xticks(tick_pos)
        ax1.set_xticklabels(tick_labels, color='#64748b', fontsize=8)

    # ===== Bottom chart: 5D/10D/20D grouped bars per strategy =====
    strat_names = list(strategies.keys())
    n_strats = len(strat_names)
    bar_width = 0.22
    x_pos = np.arange(n_strats)

    horizon_data = {h: {'avg': [], 'wr': []} for h in ['5D', '10D', '20D']}
    horizon_idx = {'5D': 4, '10D': 5, '20D': 6}
    horizon_colors = {'5D': '#60a5fa', '10D': '#34d399', '20D': '#f472b6'}

    for (name, filter_fn) in strategies.items():
        for h_name, col_idx in horizon_idx.items():
            all_rets = []
            for d in dates_sorted:
                picks = filter_fn(date_picks[d])
                for p in picks:
                    if p[col_idx] is not None:
                        all_rets.append(p[col_idx])
            if all_rets:
                horizon_data[h_name]['avg'].append(np.mean(all_rets))
                horizon_data[h_name]['wr'].append(
                    sum(1 for r in all_rets if r > 0) / len(all_rets) * 100)
            else:
                horizon_data[h_name]['avg'].append(0)
                horizon_data[h_name]['wr'].append(0)

    offsets = {'5D': -bar_width, '10D': 0, '20D': bar_width}

    for h_name in ['5D', '10D', '20D']:
        offset = offsets[h_name]
        hc = horizon_colors[h_name]
        avgs = horizon_data[h_name]['avg']
        wrs = horizon_data[h_name]['wr']

        bars = ax2.bar(x_pos + offset, avgs, bar_width * 0.9,
                       color=hc, alpha=0.85, label=h_name)

        for bar, avg_val, wr_val in zip(bars, avgs, wrs):
            if avg_val != 0:
                y_text = bar.get_height() if bar.get_height() >= 0 else bar.get_height()
                va = 'bottom' if y_text >= 0 else 'top'
                ax2.text(bar.get_x() + bar.get_width() / 2, y_text,
                         f'{avg_val:+.0f}%\n{wr_val:.0f}%',
                         ha='center', va=va, color='#e2e8f0', fontsize=6.5,
                         fontweight='bold')

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(strat_names, color='#94a3b8', fontsize=9)
    ax2.axhline(y=0, color='#475569', linewidth=0.5, linestyle='--')
    ax2.set_title('Avg Return by Horizon (with WR%)', color='#e2e8f0',
                  fontsize=11, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8, facecolor='#0f172a',
               edgecolor='#334155', labelcolor='#e2e8f0', ncol=3)
    ax2.set_ylabel('Avg Return %', color='#94a3b8', fontsize=8)

    plt.tight_layout(pad=1.5)

    # Save to base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='#0f172a', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


if __name__ == '__main__':
    b64 = generate_strategy_chart('CN', 365)
    if b64:
        print(f"✅ Generated chart: {len(b64)} bytes base64")
        import base64 as b64m
        with open('/tmp/strategy_chart_test.png', 'wb') as f:
            f.write(b64m.b64decode(b64))
        print("Saved to /tmp/strategy_chart_test.png")
