import json
import os
import requests
import logging
from datetime import datetime

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.WARNING)

# Environment variables for Lambda - define these in Lambda configuration
BASE_URL = os.environ['BASE_URL']
ORDERS_URL = "{}/v2/orders".format(BASE_URL)
ACCOUNT_URL = "{}/v2/account".format(BASE_URL)
POSITIONS_URL = "{}/v2/positions".format(BASE_URL)
AMOUNT = float(os.environ['AMOUNT'])
VALID_SIDES = {'buy', 'sell'}

# Headers for API requests - set API keys in Lambda environment variables
HEADERS = {
    'APCA-API-KEY-ID': os.environ['ALPACA_API_KEY'],
    'APCA-API-SECRET-KEY': os.environ['ALPACA_API_SECRET'],
    'Content-Type': 'application/json'
}


def create_order(symbol, qty, side, order_type, time_in_force):
    """Creates a new order in Alpaca"""
    data = {
        "symbol": symbol,
        "qty": qty,
        "side": side,
        "type": order_type,
        "time_in_force": time_in_force
    }
    r = requests.post(ORDERS_URL, json=data, headers=HEADERS)
    logger.warning(f"{datetime.now()} - {json.loads(r.content)}")
    return json.loads(r.content)


def close_position(symbol):
    """Closes all positions for a given symbol"""
    url = f"{POSITIONS_URL}/{symbol}"
    response = requests.delete(url, headers=HEADERS)
    
    if response.status_code == 200:
        logger.warning(f"{datetime.now()} - Closed all positions for {symbol}")
        return {
            "status": "success", 
            "message": f"Closed all positions for {symbol}"
        }
    else:
        logger.error(f"{datetime.now()} - Failed to close positions for {symbol}")
        logger.error(response.json())
        return {
            "status": "error", 
            "message": response.json()
        }


def get_position(symbol):
    """Checks if a position exists for a given symbol"""
    url = f"{POSITIONS_URL}/{symbol}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return response.json()  # Returns position details
    else:
        return None  # No position found


def buy(symbol, close, side="buy"):
    """Handles buy orders with position checking"""
    position = get_position(symbol)
    if position:
        response = {
            "status": "skipped", 
            "message": f"Already holding {symbol}, skipping buy order."
        }
        logger.warning(f"{datetime.now()} - Already holding {symbol}, skipping buy order.")
    else:
        qty = AMOUNT / float(close)
        response = create_order(
            symbol=symbol,
            qty=qty,
            side=side,
            order_type="market",
            time_in_force="day"
        )
        logger.warning(f"{datetime.now()} - Buy order for {symbol} placed successfully.")
    return response

def lambda_handler(event, context):
    """AWS Lambda handler function to process TradingView webhooks"""
    try:
        # Parse the incoming webhook from TradingView
        if isinstance(event.get('body'), str):
            data = json.loads(event.get('body', '{}'))
        else:
            data = event.get('body', {}) or event
        
        logger.warning(f"Received webhook: {data}")
        
        # Extract key information from the webhook
        side = data.get("side")
        symbol = data.get("symbol")
        close = data.get("close")
        
        # Skip if essential data is missing
        if not all([side, symbol, close]):
            return {
                'statusCode': 400,
                'body': json.dumps("Missing required fields in webhook payload.")
            }
        
        side = side.lower()
        if side not in VALID_SIDES:
            return {
                'statusCode': 400,
                'body': json.dumps("Invalid side provided in webhook payload.")
            }
        
        # Process the order based on side
        if side == "sell":
            response = close_position(symbol)
        else:
            response = buy(symbol, float(close), side)
        
        return {
            'statusCode': 200,
            'body': json.dumps(response)
        }
            
    except Exception as e:
        logger.error(f"Error processing webhook: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'message': 'Error processing webhook',
                'error': str(e)
            })
        }