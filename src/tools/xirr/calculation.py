import json
from datetime import datetime

from dateutil import parser
from scipy.optimize import newton


def parse_date(date_str: str) -> datetime:
    """
    Parses a date string in the format YYYY-MM-DD into a datetime object.

    Example:
    >>> parse_date("2020-01-01")
    datetime.datetime(2020, 1, 1, 0, 0)
    """
    return datetime.strptime(date_str, "%Y-%m-%d")


def calculate_xnpv(discount_rate: float, cashflows_json: str) -> float:
    """
    Calculates the Net Present Value (XNPV) given a discount rate and cashflows.

    :param discount_rate: The discount rate as a decimal (e.g., 0.1 for 10%).
    :param cashflows_json: A JSON string representing cashflows as a list of {"date": str, "amount": float}.
    :return: The calculated XNPV value.

    Example:
    >>> calculate_xnpv(0.1, '[{"date": "2020-01-01", "amount": -10000}, {"date": "2021-01-01", "amount": 2000}, {"date": "2022-01-01", "amount": 4000}]')
    1234.56
    """
    cashflows = json.loads(cashflows_json)
    cashflows = [(parse_date(entry["date"]), entry["amount"]) for entry in cashflows]

    initial_date = cashflows[0][0]
    return sum(
        amount / ((1 + discount_rate) ** ((date - initial_date).days / 365))
        for date, amount in cashflows
    )


def calculate_xirr(cashflows_json: str) -> float:
    """
    Calculates the Internal Rate of Return (XIRR) using Newton's method.

    :param cashflows_json: A JSON string representing cashflows as a list of {"date": str, "amount": float}.
    :return: The calculated XIRR value.

    Example:
    >>> calculate_xirr('[{"date": "2020-01-01", "amount": -10000}, {"date": "2021-01-01", "amount": 2000}, {"date": "2022-01-01", "amount": 4000}]')
    0.12
    """
    cashflows = json.loads(cashflows_json)
    cashflows = [(parse_date(entry["date"]), entry["amount"]) for entry in cashflows]
    initial_date = cashflows[0][0]

    def xnpv_func(rate):
        return sum(
            amount / ((1 + rate) ** ((date - initial_date).days / 365))
            for date, amount in cashflows
        )

    return newton(xnpv_func, 0.1)  # Initial guess: 10%
