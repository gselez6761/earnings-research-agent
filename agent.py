import os
from dotenv import load_dotenv
from edgar import set_identity, Company

load_dotenv()

set_identity(os.environ["EDGAR_IDENTITY"])


def get_earnings_summary(ticker: str, periods: int = 4) -> dict:
    company = Company(ticker)
    income = company.income_statement(periods=periods, annual=False)
    balance = company.balance_sheet(periods=periods, annual=False)
    return {
        "company": company.to_context(),
        "income_statement": income,
        "balance_sheet": balance,
    }


if __name__ == "__main__":
    result = get_earnings_summary("AAPL")
    print(result["company"])
    print(result["income_statement"])
