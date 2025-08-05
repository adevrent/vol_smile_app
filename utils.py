import QuantLib as ql
import datetime as dt
import numpy as np

valid_datetypes = ["ISO", "datetime", "QL"]


def convert_simple_to_ccomp(rate, tau):
    """
    Convert simple interest rate to continuously compounded rate.
    """
    if tau <= 0:
        tau = 1/365  # Default to 1 day if tau is non-positive
    return np.log(1 + rate * tau) / tau


def convert_datetype(date, to_type):
    assert to_type in valid_datetypes, "Invalid to_date type"

    # If the input is already the target type, return it.
    if to_type == "ISO" and isinstance(date, str):
        return date
    elif to_type == "datetime" and isinstance(date, dt.date) and not isinstance(date, dt.datetime):
        return date
    elif to_type == "QL" and isinstance(date, ql.Date):
        return date

    if isinstance(date, str):  # date is ISO
        assert len(date) == 10, "Date is not in valid ISO format"
        if to_type == "datetime":
            return dt.datetime.strptime(date, "%Y-%m-%d").date()
        elif to_type == "QL":
            return ql.DateParser.parseISO(date)
    elif isinstance(date, dt.date) and not isinstance(date, dt.datetime):  # date is datetime.date
        if to_type == "ISO":
            return date.strftime("%Y-%m-%d")
        elif to_type == "QL":
            return ql.Date.from_date(date)
    elif isinstance(date, dt.datetime):  # if date is datetime.datetime, convert to date first
        # Convert to date before further processing
        date_only = date.date()
        return convert_datetype(date_only, to_type)
    elif isinstance(date, ql.Date):  # date is QuantLib Date
        if to_type == "ISO":
            return date.ISO()
        elif to_type == "datetime":
            return date.to_date()

    # If none of the above conditions match, raise an error.
    raise TypeError("Unsupported date type provided.")