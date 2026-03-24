#!/usr/bin/env python3
"""Generate strategy comparison charts for email reports"""
import sqlite3, base64, io
import numpy as np
from pathlib import Path
from collections import defaultdict

V3 = Path(__file__).resolve().parent.parent

def generate_strategy_chart(market, days=60):
    """Generate strategy comparison chart as base64 PNG
    Strategies: All tiers, Small only, Large/Mega only, Mid only
    Returns base64 encoded PNG string
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

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

    # Define strategies
    if market == 'US':
        strategies = {
            '📊 All Tiers (avg)': lambda picks: picks,
            '🏢 Mega+Large': lambda picks: [p for p in picks if 'Mega' in p[1] or 'Large' in p[1]],
            '⚡ Mid': lambda picks: [p for p in picks if 'Mid' in p[1]],
            '🚀 Small+Micro': lambda picks: [p for p in picks if 'Small' in p[1] or 'Micro' in p[1]],
        }
    else:
        strategies = {
            '📊 全部 (avg)': lambda picks: picks,
            '🏢 大盘': lambda picks: [p for p in picks if '大盘' in p[1]],
            '⚡ 中盘': lambda picks: [p for p in picks if '中盘' in p[1]],
            '🚀 小盘': lambda picks: [p for p in picks if '小盘' in p[1]],
        }

    # Compute cumulative returns for each strategy
    colors = ['#818cf8', '#f59e0b', '#22c55e', '#ec4899']
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), facecolor='#0f172a',
                                     gridspec_kw={'height_ratios': [2, 1]})

    for ax in [ax1, ax2]:
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='#94a3b8', labelsize=8)
        ax.spines['bottom'].set_color('#334155')
        ax.spines['left'].set_color('#334155')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Top chart: cumulative return
    for (name, filter_fn), color in zip(strategies.items(), colors):
        cum_ret = [0]
        for d in dates_sorted:
            picks = filter_fn(date_picks[d])
            if picks:
                # Use best available return
                rets = []
                for p in picks:
                    r = p[6] if p[6] is not None else p[5] if p[5] is not None else p[4]
                    if r is not None:
                        rets.append(r)
                if rets:
                    cum_ret.append(cum_ret[-1] + np.mean(rets))
                else:
                    cum_ret.append(cum_ret[-1])
            else:
                cum_ret.append(cum_ret[-1])

        x = range(len(cum_ret))
        ax1.plot(x, cum_ret, color=color, linewidth=2, label=name, alpha=0.9)
        ax1.fill_between(x, 0, cum_ret, color=color, alpha=0.08)

    ax1.axhline(y=0, color='#475569', linewidth=0.5, linestyle='--')
    ax1.set_title(f'{"US" if market == "US" else "CN"} 策略累计收益 (最近{days}天)',
                  color='#e2e8f0', fontsize=12, fontweight='bold', pad=10)
    ax1.set_ylabel('累计收益 %', color='#94a3b8', fontsize=9)
    ax1.legend(loc='upper left', fontsize=8, facecolor='#0f172a',
               edgecolor='#334155', labelcolor='#e2e8f0')
    ax1.yaxis.set_major_formatter(PercentFormatter(decimals=0))

    # Bottom chart: per-strategy bar (win rate + avg return)
    strat_names = []
    strat_avg = []
    strat_wr = []
    strat_colors = []

    for (name, filter_fn), color in zip(strategies.items(), colors):
        all_rets = []
        for d in dates_sorted:
            picks = filter_fn(date_picks[d])
            for p in picks:
                r = p[6] if p[6] is not None else p[5] if p[5] is not None else p[4]
                if r is not None:
                    all_rets.append(r)
        if all_rets:
            strat_names.append(name.split(' ')[-1])
            strat_avg.append(np.mean(all_rets))
            strat_wr.append(sum(1 for r in all_rets if r > 0) / len(all_rets) * 100)
            strat_colors.append(color)

    if strat_names:
        x = np.arange(len(strat_names))
        bars = ax2.bar(x - 0.2, strat_avg, 0.35, color=strat_colors, alpha=0.8, label='Avg Return %')
        bars2 = ax2.bar(x + 0.2, strat_wr, 0.35, color=strat_colors, alpha=0.4, label='Win Rate %')

        for bar, val in zip(bars, strat_avg):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:+.1f}%', ha='center', va='bottom', color='#e2e8f0', fontsize=7)
        for bar, val in zip(bars2, strat_wr):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.0f}%', ha='center', va='bottom', color='#94a3b8', fontsize=7)

        ax2.set_xticks(x)
        ax2.set_xticklabels(strat_names, color='#94a3b8', fontsize=8)
        ax2.axhline(y=0, color='#475569', linewidth=0.5, linestyle='--')
        ax2.set_title('各策略表现', color='#e2e8f0', fontsize=10, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=7, facecolor='#0f172a',
                   edgecolor='#334155', labelcolor='#e2e8f0')

    plt.tight_layout(pad=1.5)

    # Save to base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='#0f172a', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


if __name__ == '__main__':
    b64 = generate_strategy_chart('US', 60)
    if b64:
        print(f"✅ Generated chart: {len(b64)} bytes base64")
        # Save test image
        import base64 as b64m
        with open('/tmp/strategy_chart_test.png', 'wb') as f:
            f.write(b64m.b64decode(b64))
        print("Saved to /tmp/strategy_chart_test.png")
