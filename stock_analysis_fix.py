
from Stock_utils.stock_analysis import StockAnalysis
import pandas as pd
import numpy as np

def patch_stock_analysis():
    """Monkey-patch the StockAnalysis class to add LIRED calculation"""
    
    # Save the original method
    original_calculate_phantom_force = StockAnalysis.calculate_phantom_force
    
    # Define the new method that adds LIRED
    def new_calculate_phantom_force(self):
        # Call the original method
        df = original_calculate_phantom_force(self)
        
        # Add LIRED calculation
        try:
            if 'phantom_pink' in df.columns and 'phantom_blue' in df.columns:
                df['phantom_lired'] = df['phantom_pink'] - df['phantom_blue']
        except Exception as e:
            print(f"Error adding LIRED calculation: {e}")
        
        return df
    
    # Replace the original method with our new one
    StockAnalysis.calculate_phantom_force = new_calculate_phantom_force
    
    # Also add the standalone LIRED calculation method
    def calculate_phantom_lired(self):
        if 'phantom_pink' not in self.df.columns or 'phantom_blue' not in self.df.columns:
            self.df = self.calculate_phantom_force()
        
        try:
            self.df['phantom_lired'] = self.df['phantom_pink'] - self.df['phantom_blue']
        except Exception as e:
            print(f"Error in standalone LIRED calculation: {e}")
            
        return self.df
    
    # Add the new method to the class
    StockAnalysis.calculate_phantom_lired = calculate_phantom_lired
    