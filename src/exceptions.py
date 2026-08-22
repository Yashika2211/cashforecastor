class CashflowError(Exception):
    """Base error for the forecasting pipeline."""


class InvalidLedgerError(CashflowError):
    pass


class InsufficientHistoryError(CashflowError):
    def __init__(self, merchant_id: str, available_days: int, required_days: int):
        self.merchant_id = merchant_id
        self.available_days = available_days
        self.required_days = required_days
        super().__init__(
            f"Merchant {merchant_id} has {available_days} daily rows; "
            f"need at least {required_days}."
        )


class ModelNotTrainedError(CashflowError):
    def __init__(self):
        super().__init__("Forecast model has not been trained yet.")


class MerchantNotFoundError(CashflowError):
    def __init__(self, merchant_id: str):
        self.merchant_id = merchant_id
        super().__init__(f"No ledger history for merchant {merchant_id}.")
