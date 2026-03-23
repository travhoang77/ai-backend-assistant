from datetime import datetime

def get_current_time():
    return str(datetime.now())

def calculate_investment(amount, rate, years):
    if rate > 1:
        rate = rate / 100
    return amount * (1 + rate) ** years