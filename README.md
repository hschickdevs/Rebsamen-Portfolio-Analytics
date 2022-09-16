1. Figure the models/valuation method
2. Test the models
3. Implement models into user interface

# Option 1: Web dashboard that shows the following:
https://hschickdevs-rebsamen-portfolio-analytics-dashboard-qio5q1.streamlitapp.com/

* Can use this: https://github.com/gerardrbentley/fidelity-account-overview
* Buy or sell recommendations can be produced for each stock based on predetermined models
* Positions should be able to be marked as hold, sell, or buy
* Could implement a watchlist feature

_**Components (Top-Down)**_:
1. Rebsamen Portfolio Overview Header
2. "How to Use This" documentation dropdown
3. Rebsamen Holdings CSV Upload Section
    - Drag or Drop File here
    - Clean data dropdown
4. Holdings Table
    - Shows all holdings with ticker data and valuations
5. Total Rebsamen Trust Value Section
    - Value by Total Account (incl. change since inception)
    - Value by Asset Class (i.e. EQUITY, CASH, MONEY FUNDS, BANK DEPOSITS)

**Resources:**
- https://valueinvesting.io/
- https://medium.datadriveninvestor.com/creating-an-automated-stock-screener-in-notion-with-python-43df78293bb4
- https://equityapi.morningstar.com/
- https://github.com/marcosan93/Medium-Misc-Tutorials/blob/main/Stock-Market-Tutorials/Analyze-Fundamental-Data.ipynb

# Option 2: Telegram bot with the following features:
- Resources:
    * [Streamlit](https://streamlit.io/)
    * Dash
    * Voilà
    * [Panel](https://panel.holoviz.org/)
- Send either a single ticker or a list of tickers to run the model on
- Returns automated excel spreadsheets for the model in a zip file
- Run in Linode ssh server

# Valuation Notes:
- Calculate target price by weighted valuations (numbered by priority)
    * (#1) ENTERPRISE VALUE / EBIDTA 50% weight:
      * AKA Enterprise multiple, also known as the EV-to-EBITDA multiple, is a ratio used to determine the value of a company.
      * It is computed by dividing enterprise value by EBITDA.
      * The enterprise multiple takes into account a company's debt and cash levels in addition to its stock price and relates that value to the firm's cash profitability. Enterprise multiples can vary depending on the industry.
      * Higher enterprise multiples are expected in high-growth industries and lower multiples in industries with slow growth
      
    * (#2) PRICE / EBIDTA:
  
    * (#3) PRICE / EARNINGS:
  
    * DCF Estimate (not being used in the competition):
  
    * PRICE / SALES:
  
    * PRICE / BOOK:
  
- Target price valuation weights should be adjustable by the user
- Utilize insider sentiment and transactions
    * Sheet in the excel file that shows all insider trades in the relevant timeframe