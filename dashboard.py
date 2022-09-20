import functools
from pathlib import Path
import mimetypes
from typing import Optional, Union
import time
from math import floor
from typing import Optional, Union

from src.finnhub_client import FinnhubClient

import streamlit as st
from st_aggrid import AgGrid, ColumnsAutoSizeMode
from st_aggrid.shared import JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder
import pandas as pd
import plotly.express as px
import numpy as np

finnhub_client = FinnhubClient(st.secrets["FINNHUB_API_KEY"])


chart = functools.partial(st.plotly_chart, use_container_width=True)
COMMON_ARGS = {
    "color": "Security ID",
    "color_discrete_sequence": px.colors.sequential.Greens,
    "hover_data": [
        "Asset Classification",
        "Sub-Asset Classification ",
        "Quantity",
        "Price",
        "Price as of date",
    ]
}
# PIE_CHART_ARGS = {
#     # "color": "Security ID",
#     # "color_discrete_sequence": px.colors.sequential.Greens,
#     "hover_data": [
#         "Asset Classification",
#         "Sub-Asset Classification ",
#         "Quantity",
#         "Price",
#         "Price as of date",
#     ]
# }

@st.experimental_memo
def get_company_basic_financials(ticker: str, holdings_date: str) -> tuple[str, Union[pd.DataFrame, None]]:
    """
    Get company basic financials from finnhub api

    Args:
        ticker (str): ticker symbol
        holdings_date (str): date of holdings (for performance)

    Returns:
        str: The company symbol/ticker
        pd.DataFrame: company basic financials
    """
    holdings_date = holdings_date  # Cache previous calls for this holding date to preserve API calls
    r = finnhub_client.company_basic_financials(ticker, "all")
    print("Hit Finnhub API for {symbol} at {date} basic financials".format(symbol=ticker, date=holdings_date))
    if len(r) > 0:
            # json.dumps(r, indent=3)
            formatted_r = {"metric": [k for k, v in r['metric'].items()],
                            "value": [str(v) for k, v in r['metric'].items()]}
            formatted_df = pd.DataFrame(formatted_r)
            return ticker, formatted_df
    else:
        return ticker, None


@st.experimental_memo
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Take Raw Fidelity Dataframe and return usable dataframe.
    - snake_case headers
    - Include 401k by filling na type
    - Drop Cash accounts and misc text
    - Clean $ and % signs from values and convert to floats

    Args:
        df (pd.DataFrame): Raw fidelity csv data

    Returns:
        pd.DataFrame: cleaned dataframe with features above
    """
    df = df.copy()
    df.columns = df.columns.str.lower().str.replace(" ", "_", regex=False).str.replace("/", "_", regex=False)

    df.type = df.type.fillna("unknown")
    df = df.dropna()

    price_index = df.columns.get_loc("last_price")
    cost_basis_index = df.columns.get_loc("cost_basis_per_share")
    df[df.columns[price_index : cost_basis_index + 1]] = df[
        df.columns[price_index : cost_basis_index + 1]
    ].transform(lambda s: s.str.replace("$", "", regex=False).str.replace("%", "", regex=False).astype(float))

    quantity_index = df.columns.get_loc("quantity")
    most_relevant_columns = df.columns[quantity_index : cost_basis_index + 1]
    first_columns = df.columns[0:quantity_index]
    last_columns = df.columns[cost_basis_index + 1 :]
    df = df[[*most_relevant_columns, *first_columns, *last_columns]]
    return df


@st.experimental_memo
def clean_rebsamen_data(df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    """
    Take Raw Rebsamen Trust Holdings Dataframe and return usable dataframe.
    - Remove top 2 rows and use 3rd row as header

    Args:
        df (pd.DataFrame): Raw rebsamen trust holdings .xls data

    Returns:
        pd.DataFrame: cleaned dataframe with features above
    """
    if sheet_name.lower() == "positions":
        # 1. Fix header
        updated_header = df.iloc[1] #grab the second row for the header
        df = df[2:]
        df.columns = updated_header
        df = df.drop(df[df['Quantity'] == 0].index)  # Remove empty positions (0 qty)
    elif sheet_name.lower() in ["valuation over time", "VOT"]:
        updated_header = df.iloc[0] #grab the second row for the header
        df = df[1:]
        df.columns = updated_header
    elif sheet_name.lower() == "realized gains":
        pass
    elif sheet_name.lower() == "unrealized gains":
        pass
    
    # Clear the cache for the basic financials function since we are using a new sheet
    get_company_basic_financials.clear()
    
    return df

@st.cache
def convert_df(df):
    return df.to_csv().encode('utf-8')

@st.experimental_memo
def load_rebsamen_data(file: Union[str, bytes]) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    """
    The load_rebsamen_data function loads the positions and valuation data from the Rebsamen portfolio report.
    The function returns the holdings date, positions dataframe, and valuation dataframe.
    This function is used to memoize the dataframes so that they are not reloaded on every page refresh.
    
    :param file: Allow the function to accept either a string or bytes object
    :return: A tuple 
        - Holdings Date (str)
        - Rebsamen Positions Data (pd.DataFrame)
        - Rebsamen VOT Data (pd.DataFrame)
    """
    rebsamen_positions_data = pd.read_excel(file, sheet_name="Positions")
    holdings_date = rebsamen_positions_data.columns[0].split(' ')[-1]
    rebsamen_positions_data = clean_rebsamen_data(rebsamen_positions_data, "Positions")
    
    # TODO: Fix this
    try:
        rebsamen_valuation_data = clean_rebsamen_data(pd.read_excel(file, sheet_name="Valuation over Time"), sheet_name="Valuation over Time")
    except ValueError:
        # Temporary fix for different sheet names
        rebsamen_valuation_data = clean_rebsamen_data(pd.read_excel(file, sheet_name="VOT"), sheet_name="VOT")
        
    return holdings_date, rebsamen_positions_data, rebsamen_valuation_data
    

@st.experimental_memo
def filter_data(
    df: pd.DataFrame, account_selections: list[str], symbol_selections: list[str]
) -> pd.DataFrame:
    """
    Returns Dataframe with only accounts and symbols selected

    Args:
        df (pd.DataFrame): clean fidelity csv data, including account_name and symbol columns
        account_selections (list[str]): list of account names to include
        symbol_selections (list[str]): list of symbols to include

    Returns:
        pd.DataFrame: data only for the given accounts and symbols
    """
    df = df.copy()
    df = df[
        df.account_name.isin(account_selections) & df.symbol.isin(symbol_selections)
    ]

    return df

@st.experimental_memo
def get_company_profile(ticker: str, holdings_date: str) -> dict:
    """
    Get company profile from IEX Cloud API

    Args:
        ticker (str): ticker symbol
        holdings_date (str): date of holdings from Rebsamen report (allows for correct caching)

    Returns:
        dict: company profile
    """
    # holdings_date = holdings_date
    return finnhub_client.get_company_profile2(ticker)


def main() -> None:
    #---------------------------SIDEBAR (SETTINGS IN SIDE PANEL) SETUP MENU---------------------------------------"""
    st.sidebar.header("Dashboard Settings:")
    
    # UPLOAD 
    # st.sidebar.subheader("Upload the Rebsamen Holdings .xls file:")
    uploaded_data = st.sidebar.file_uploader(
        "Upload the Rebsamen holdings file", type=[".csv", ".xls", ".xlsx", ".xlsm"], accept_multiple_files=False
    )
    
    # File upload setup
    if uploaded_data is None:
        st.sidebar.info("Using sample data. Upload a file above to use your own data!")
        uploaded_data = open(Path('src/assets/example.xls'), "rb")
    else:
        st.sidebar.success("Uploaded your file!")

    
    # REBSAMEN DATA FOR TESTING:
    # rebsamen_positions_data = pd.read_excel(uploaded_data, sheet_name="Positions")
    # holdings_date = rebsamen_positions_data.columns[0].split(' ')[-1]
    # rebsamen_positions_data = clean_rebsamen_data(rebsamen_positions_data, "Positions")
    
    holdings_date, rebsamen_positions_data, rebsamen_valuation_data = load_rebsamen_data(uploaded_data)
    
    # LOADING HEADER TO PROCESS POSITIONS DATA
    loading_header = st.header("Processing data...")
    loading_bar = st.progress(0.0)
    industries = []
    logos = []
    increment = 100 / len(rebsamen_positions_data)
    for i, row in rebsamen_positions_data.iterrows():
        try:
            loading_bar.progress((increment * (i - 1)) / 100)
        except:
            # TODO: Temp fix for invalid progress value error
            pass
        loading_header.header(f"Processing data... {floor(increment * (i - 1))}%")
        
        # Get company descriptions
        r = get_company_profile(row['Security ID'], holdings_date)
        if len(r) > 0:
            industries.append(r['finnhubIndustry'])
            logos.append(r['logo'])
        else:
            industries.append("Other")
            logos.append(None)
    
    rebsamen_positions_data['Industry'] = industries
    rebsamen_positions_data['Logo'] = logos
        
    loading_bar.empty()
    loading_header.empty()
    ###############################################################################################################"""
    
    # TODO: Fix this
    # try:
    #     rebsamen_valuation_data = clean_rebsamen_data(pd.read_excel(uploaded_data, sheet_name="Valuation over Time"), sheet_name="Valuation over Time")
    # except ValueError:
    #     # Temporary fix for different sheet names
    #     rebsamen_valuation_data = clean_rebsamen_data(pd.read_excel(uploaded_data, sheet_name="VOT"), sheet_name="VOT")
    
    # rebsamen_realized_gains_data = pd.read_excel(Path("src/assets/Rebsamen Holdings 082222.xls"), sheet_name="Realized Gains")
    # rebsamen_unrealized_gains_data = pd.read_excel(Path("src/assets/Rebsamen Holdings 082222.xls"), sheet_name="Unrealized Gains")
        
    st.sidebar.download_button(label="Download Current Data", data=uploaded_data.name, file_name=uploaded_data.name, mime=mimetypes.guess_type(uploaded_data.name)[0])
    
    st.sidebar.header("Valuation Assumptions:")
    st.sidebar.write("*Work in progress...*")
        
    # raw_df = pd.read_csv(uploaded_data)
    # with st.expander("Raw Data"):
    #     st.write(raw_df)

    # df = clean_data(raw_df)
    # with st.expander("Cleaned Data"):
    #     st.write(rebsamen_positions_data)
        
    
    # st.sidebar.subheader("Filter Displayed Accounts")

    # accounts = list(df.account_name.unique())
    # account_selections = ["PLACEHOLDER"]
    # account_selections = st.sidebar.multiselect(
    #     "Select Accounts to View", options=accounts, default=accounts
    # )
    # st.sidebar.subheader("Filter Displayed Tickers")

    # symbols = list(df.loc[df.account_name.isin(account_selections), "symbol"].unique())
    # symbol_selections = ["PLACEHOLDER"]
    # symbol_selections = st.sidebar.multiselect(
    #     "Select Ticker Symbols to View", options=symbols, default=symbols
    # )
    #--------------------------------------------------------------------------------------------------------------"""
    
    
    #-------------------------------------HEADER AND TUTORIAL---------------------------------------------------- """
    st.header(f"Rebsamen Portfolio (as of {holdings_date})")

    # with st.expander("How to Use"):
    #     st.write(Path("assets/placeholder.txt").read_text())
    #--------------------------------------------------------------------------------------------------------------"""


    #---------------------------------SELECTED ACCOUNT AND TICKER DATA CHART---------------------------------------"""
    # df = filter_data(df, account_selections, symbol_selections)
    # st.subheader("Selected Account and Ticker Data")
    # cellsytle_jscode = JsCode(
    #     """
    # function(params) {
    #     if (params.value > 0) {
    #         return {
    #             'color': 'white',
    #             'backgroundColor': 'forestgreen'
    #         }
    #     } else if (params.value < 0) {
    #         return {
    #             'color': 'white',
    #             'backgroundColor': 'crimson'
    #         }
    #     } else {
    #         return {
    #             'color': 'white',
    #             'backgroundColor': 'slategray'
    #         }
    #     }
    # };
    # """
    # )

    # gb = GridOptionsBuilder.from_dataframe(df)
    # gb.configure_columns(
    #     (
    #         "last_price_change",
    #         "total_gain_loss_dollar",
    #         "total_gain_loss_percent",
    #         "today's_gain_loss_dollar",
    #         "today's_gain_loss_percent",
    #     ),
    #     cellStyle=cellsytle_jscode,
    # )
    # gb.configure_pagination()
    # gb.configure_columns(("account_name", "symbol"), pinned=True)
    # gridOptions = gb.build()

    # AgGrid(df, gridOptions=gridOptions, allow_unsafe_jscode=True)

    # def draw_bar(y_val: str) -> None:
    #     fig = px.bar(df, y=y_val, x="symbol", **COMMON_ARGS)
    #     fig.update_layout(barmode="stack", xaxis={"categoryorder": "total descending"})
    #     chart(fig)
    #--------------------------------------------------------------------------------------------------------------"""


    # ----------------------------------------- ACCOUNT VALUES & CHARTS -------------------------------------------"""
    # st.subheader(f'Portfolio Value')
    # totals = df.groupby("account_name", as_index=False).sum()
    
    asset_classes = rebsamen_positions_data.groupby("Asset Classification")
    
    ytd_return = sum(filter(lambda i: not (type(i) is str), list(rebsamen_valuation_data[rebsamen_valuation_data.columns[7]])))
    print(ytd_return)
    # ytd_return = sum([v for v in list(rebsamen_valuation_data[rebsamen_valuation_data.columns[7]]) if v is not str])
    st.metric(
        "Total Portfolio Value",
        f'${rebsamen_positions_data["Market Value"].sum():,.2f}',
        f"{ytd_return:,.2f} YTD",
    )
    # for column, row in zip(st.columns(len(totals)), totals.itertuples()):
    #     column.metric(
    #         row.account_name,
    #         f"${row.current_value:.2f}",
    #         f"{row.total_gain_loss_dollar:.2f}",
    #     )
    for column, group in zip(st.columns(len(asset_classes)), reversed(tuple(asset_classes))):
        column.metric(
            group[0].title(),
            f"${group[1]['Market Value'].sum():,.2f}",
            # f"{row.total_gain_loss_dollar:.2f}",
        )

    # fig = px.bar(
    #     totals,
    #     y="account_name",
    #     x="current_value",
    #     color="account_name",
    #     color_discrete_sequence=px.colors.sequential.Greens,
    # )
    # fig.update_layout(barmode="stack", xaxis={"categoryorder": "total descending"})
    # chart(fig)

    st.subheader("Value by Symbol")
    # draw_bar("current_value")
    
    fig = px.bar(rebsamen_positions_data, y="Market Value", x="Security ID", **COMMON_ARGS)
    fig.update_layout(barmode="stack", xaxis={"categoryorder": "total descending"})
    chart(fig)
    
    st.subheader("Value Composition by Symbol & Industry Sector")
    vc_col1, vc_col2 = st.columns(2)
    with vc_col1:
        
        fig = px.pie(rebsamen_positions_data, values="Market Value", names="Security ID", color="Security ID", color_discrete_sequence=px.colors.sequential.Greens)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        chart(fig)
    with vc_col2:
        fig = px.pie(rebsamen_positions_data, values="Market Value", names="Industry", color="Industry", color_discrete_sequence=px.colors.sequential.Greens)
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        chart(fig)
        
    st.subheader("Financials & Valuation")
    # Get company fundamentals, maybe use loading indicators here 
    
    ###TICKER SELECTBOX EXPERIMENTATION"""
    ticker = st.selectbox("Select a Ticker from the Portfolio:", list(rebsamen_positions_data["Security ID"].sort_values(ascending=True).unique()))
    row = rebsamen_positions_data.loc[rebsamen_positions_data['Security ID'] == ticker.upper()]
    with st.spinner(f"Loading {ticker}..."):
        # print(row)
        if len(row) > 0:
            if row['Logo'].iat[0] is not None:    
                st.image(row['Logo'].iat[0], width=100)
            st.subheader(f"({row['Security ID'].iat[0]}) {row['Description'].iat[0]}")
            # tab.write(f"**Last Price:** \${row[1]['Price']} (as of {row[1]['Price as of date']} {row[1]['Timezone']})  |  **Shares:** {row[1]['Quantity']}  |  **NAV:** ${row[1]['Market Value']:,.2f}")
            st.write(f"**Position Value:** ${row['Market Value'].iat[0]:,.2f}")
            st.write(f"**Shares:** {row['Quantity'].iat[0]}")
            st.write(f"**Last Price:** \${row['Price'].iat[0]} (as of {row['Price as of date'].iat[0]} {row['Timezone'].iat[0]})")
            
            for sub_i, sub_tab in enumerate(st.tabs(["🏦 Valuation", "📊 Financial Metrics", "📈 Technicals"])):
                if sub_i == 0:
                    with sub_tab.expander("Algorithmic Stock Rating"):
                        st.write("This is a placeholder for the algorithmic stock rating result")
                    sub_tab.write("Select valuation method:")
                    for v_tab_i, v_tab in enumerate(sub_tab.tabs(['DCF', 'Multiples Analysis'])):
                        v_tab.write(f"This is a placeholder for the valuation tab")
                elif sub_i == 1:
                    # r =  finnhub_client.company_basic_financials(row[1]['Security ID'], "all")
                    symbol, formatted_df = get_company_basic_financials(row['Security ID'].iat[0], holdings_date)
                    if formatted_df is not None:
                        # formatted_r = {"metric": [k for k, v in r['metric'].items()],
                        #                 "value": [str(v) for k, v in r['metric'].items()]}
                        # formatted_df = pd.DataFrame(formatted_r)
                        with sub_tab:
                            AgGrid(formatted_df, key=f"grid-{symbol}", fit_columns_on_grid_load=True, columns_auto_size_mode=ColumnsAutoSizeMode.FIT_ALL_COLUMNS_TO_VIEW)  # fit_columns_on_grid_load=True, 
                        sub_tab.download_button(
                            label="Download Metrics as CSV",
                            data=convert_df(formatted_df),
                            file_name=f'{symbol}_financial_metrics.csv',
                            mime='text/csv',
                            key=f"download-{symbol}",
                        )
                    else:
                        sub_tab.write("No financials available")
                elif sub_i == 2:
                    sub_tab.image(f"https://finviz.com/chart.ashx?t={symbol.upper().strip()}&ty=c&ta=1&p=d&s=l")
                    sub_tab.write("*Proof of Concept - Not a real chart*")
        else:
            st.write(f"Invalid symbol: {ticker}")
    ###END EXPERIMENTATION"""
    

if __name__ == "__main__":
    st.set_page_config(
        "Rebsamen Analytics",
        "📊",
        initial_sidebar_state="expanded",
        layout="wide",
    )
    main()