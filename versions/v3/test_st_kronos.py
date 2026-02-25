import streamlit as st
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml.data_cache import DataCache

st.title("Test Kronos")
symbol = st.text_input("Symbol", "HIMS")

@st.cache_resource
def load_engine():
    import ml.kronos_integration as ki
    return ki.get_kronos_engine()

if st.button("Predict"):
    engine = load_engine()
    cache = DataCache()
    df = cache.get_stock_history(symbol, market="US")
    if df is not None:
        st.write(f"Loaded {len(df)} rows")
        df_input = df.copy()
        df_input.rename(columns={'date': 'timestamps', 'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close', 'volume': 'volume'}, inplace=True)
        df_input.columns = [c.lower() for c in df_input.columns]
        if 'timestamps' not in df_input.columns and 'date' in df_input.columns:
            df_input.rename(columns={'date': 'timestamps'}, inplace=True)
        df_input = df_input.tail(400)
        
        with st.spinner("Predicting..."):
            pred = engine.predict_future_klines(df_input, pred_len=20, temperature=0.5, top_p=0.8)
            if pred is not None:
                st.write("Done!")
                st.write(pred.head())
            else:
                st.write("Fail")
