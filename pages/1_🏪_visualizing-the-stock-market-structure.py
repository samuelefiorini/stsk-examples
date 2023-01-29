import numpy as np
import pandas as pd
import streamlit as st
from sklearn import covariance

SYMBOLS_DICT = {
    "TOT": "Total",
    "XOM": "Exxon",
    "CVX": "Chevron",
    "COP": "ConocoPhillips",
    "VLO": "Valero Energy",
    "MSFT": "Microsoft",
    "IBM": "IBM",
    "TWX": "Time Warner",
    "CMCSA": "Comcast",
    "CVC": "Cablevision",
    "YHOO": "Yahoo",
    "DELL": "Dell",
    "HPQ": "HP",
    "AMZN": "Amazon",
    "TM": "Toyota",
    "CAJ": "Canon",
    "SNE": "Sony",
    "F": "Ford",
    "HMC": "Honda",
    "NAV": "Navistar",
    "NOC": "Northrop Grumman",
    "BA": "Boeing",
    "KO": "Coca Cola",
    "MMM": "3M",
    "MCD": "McDonald's",
    "PEP": "Pepsi",
    "K": "Kellogg",
    "UN": "Unilever",
    "MAR": "Marriott",
    "PG": "Procter Gamble",
    "CL": "Colgate-Palmolive",
    "GE": "General Electrics",
    "WFC": "Wells Fargo",
    "JPM": "JPMorgan Chase",
    "AIG": "AIG",
    "AXP": "American express",
    "BAC": "Bank of America",
    "GS": "Goldman Sachs",
    "AAPL": "Apple",
    "SAP": "SAP",
    "CSCO": "Cisco",
    "TXN": "Texas Instruments",
    "XRX": "Xerox",
    "WMT": "Wal-Mart",
    "HD": "Home Depot",
    "GSK": "GlaxoSmithKline",
    "PFE": "Pfizer",
    "SNY": "Sanofi-Aventis",
    "NVS": "Novartis",
    "KMB": "Kimberly-Clark",
    "R": "Ryder",
    "GD": "General Dynamics",
    "RTN": "Raytheon",
    "CVS": "CVS",
    "CAT": "Caterpillar",
    "DD": "DuPont de Nemours",
}

URL = (
    "https://raw.githubusercontent.com/scikit-learn/examples-data/"
    "master/financial-data/{}.csv"
)


@st.experimental_memo
def download_data(selected: list) -> pd.DataFrame:
    """Download selected symbols."""
    symbols_dict = dict(filter(lambda x: x[1] in selected, SYMBOLS_DICT.items()))
    symbols, names = np.array(sorted(symbols_dict.items())).T
    bar, quotes = st.progress(0), []
    for i, symbol in enumerate(symbols):
        quotes.append(pd.read_csv(URL.format(symbol)))
        bar.progress((i + 1) / len(selected))
    return quotes, names


# Init session states
_ = st.session_state.setdefault("quotes", None)
_ = st.session_state.setdefault("names", None)
_ = st.session_state.setdefault("model", None)

# Main
st.title("Visualizing the stock market structure")

with st.expander("Credits", expanded=False):
    st.caption(
        "Click [here](https://scikit-learn.org/stable/auto_examples/applications/plot_stock_market.html#sphx-glr-auto-examples-applications-plot-stock-market-py) to see the original code."
    )
    st.code(
        """
# Author: Gael Varoquaux gael.varoquaux@normalesup.org
# License: BSD 3 clause
        """
    )

st.write(
    """
    This example employs several unsupervised learning techniques
    to extract the stock market structure from variations in historical quotes.

    The quantity that we use is the daily variation in quote price:
    quotes that are linked tend to fluctuate in relation to each other during a day.
    """
)

if st.sidebar.button("♻️ Clear"):
    st.experimental_memo.clear()
    st.experimental_rerun()

st.markdown("## Retrieve the data from Internet")
st.write(
    """
    The data is from 2003 - 2008.
    This is reasonably calm: (not too long ago so that we get high-tech firms, and before the 2008 crash).
    This kind of historical data can be obtained from APIs like the [data.nasdaq.com](https://data.nasdaq.com/) and [alphavantage.co](alphavantage.co).
    """
)
with st.expander("Symbols", expanded=False):
    symbols = pd.Series(SYMBOLS_DICT).sort_values()
    default = symbols if st.checkbox("Select all") else []
    selected = st.multiselect("Symbols", symbols, default=default)

    if st.button("⬇️ Download"):
        quotes, names = download_data(selected)
        st.session_state.quotes = quotes
        st.session_state.names = names

if st.session_state.quotes is not None:
    close_prices = np.vstack([q["close"] for q in st.session_state.quotes])
    open_prices = np.vstack([q["open"] for q in st.session_state.quotes])

    # The daily variations of the quotes are what carry the most information
    variation = close_prices - open_prices

    st.markdown("## Learning the graph structure")
    st.write(
        """
        We use sparse inverse covariance estimation to find which quotes are correlated conditionally on the others.
        Specifically, sparse inverse covariance gives us a graph, that is a list of connections.
        For each symbol, the symbols that it is connected to are those useful to explain its fluctuations
        """
    )
    with st.expander("Model"):
        cols = st.columns(3)
        with cols[0]:
            alpha_min = st.number_input(
                "α min", min_value=-10.0, max_value=10.0, value=-1.5
            )
        with cols[1]:
            alpha_max = st.number_input(
                "α max", min_value=alpha_min, max_value=10.0, value=1.0
            )
        with cols[2]:
            n_alphas = st.number_input("N°α", min_value=1, max_value=25, value=10)
        cols = st.columns(2)
        with cols[0]:
            alphas = np.logspace(alpha_min, alpha_max, num=10)
            edge_model = covariance.GraphicalLassoCV(alphas=alphas)
            # standardize the time series: using correlations rather than covariance
            # former is more efficient for structure recovery
            X = variation.copy().T
            X /= X.std(axis=0)
            st.markdown("<br/>", unsafe_allow_html=True)
            if st.button("🚀 Fit"):
                with st.spinner("..."):
                    edge_model.fit(X)
                    st.session_state.model = edge_model
        with cols[1]:
            st.markdown("<br/>", unsafe_allow_html=True)
            if st.button("💣 Reset"):
                st.session_state.model = None

if st.session_state.model is not None:
    st.markdown("## Clustering using affinity propagation")
    st.write(
        """
        We use clustering to group together quotes that behave similarly.
        Here, amongst the [various clustering techniques](https://scikit-learn.org/stable/modules/clustering.html#clustering)
        available in the scikit-learn, we use [Affinity Propagation](https://scikit-learn.org/stable/modules/clustering.html#affinity-propagation) as it does not enforce equal-size clusters,
        and it can choose automatically the number of clusters from the data.

        Note that this gives us a different indication than the graph, as the graph
        reflects conditional relations between variables, while the clustering reflects
        marginal properties: variables clustered together can be considered as having a
        similar impact at the level of the full stock market.
        """
    )
