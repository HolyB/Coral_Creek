import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from db.database import insert_scan_result

result = {
    'Symbol': 'TEST',
    'Date': '2026-01-08',
    'Price': 100,
    'Turnover_M': 100,
    'Blue_Daily': 100,
    'Blue_Weekly': 100,
    'Blue_Monthly': 100,
    'ADX': 100,
    'Volatility': 100,
    'Is_Heima': False,
    'Is_Juedi': False,
    'Heima_Daily': False,
    'Heima_Weekly': False,
    'Heima_Monthly': False,
    'Juedi_Daily': False,
    'Juedi_Weekly': False,
    'Juedi_Monthly': False,
    'Strat_D_Trend': True,
    'Strat_C_Resonance': False,
    'Legacy_Signal': False,
    'Regime': 'Standard',
    'Adaptive_Thresh': 100,
    'VP_Rating': 'Normal',
    'Profit_Ratio': 0.5,
    'Wave_Phase': 'Wave1',
    'Wave_Desc': 'Desc',
    'Chan_Signal': 'Signal',
    'Chan_Desc': 'Desc',
    'Duokongwang_Buy': False,
    'Duokongwang_Sell': False,
    'Market_Cap': 100,
    'Cap_Category': 'Micro',
    'Company_Name': 'TEST',
    'Industry': 'TEST',
    'Day_High': 100,
    'Day_Low': 100,
    'Day_Close': 100,
    'Stop_Loss': 90,
    'Shares_Rec': 10,
    'Risk_Reward_Score': 100,
    'Market': 'US',
    'ML_Rank_Score': 100
}

try:
    insert_scan_result(result)
    print("Success")
except Exception as e:
    import traceback
    traceback.print_exc()
