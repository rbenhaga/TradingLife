"""Utility functions for trading calculations"""

def calculate_position_size(balance: float, risk_percentage: float, entry_price: float, stop_loss: float) -> float:
    """
    Calculate the position size based on risk management parameters.
    
    Args:
        balance (float): Account balance
        risk_percentage (float): Maximum risk percentage (0-100)
        entry_price (float): Entry price of the position
        stop_loss (float): Stop loss price
        
    Returns:
        float: Position size in base currency
    """
    risk_amount = balance * (risk_percentage / 100)
    price_risk = abs(entry_price - stop_loss)
    position_size = risk_amount / price_risk
    return position_size

def format_number(number: float, decimals: int = 8) -> str:
    """Format a number with specified decimal places"""
    return f"{number:.{decimals}f}"

def calculate_risk_reward_ratio(entry_price: float, stop_loss: float, take_profit: float) -> float:
    """Calculate the risk/reward ratio of a trade"""
    risk = abs(entry_price - stop_loss)
    reward = abs(take_profit - entry_price)
    return reward / risk if risk > 0 else 0 