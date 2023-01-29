import numpy as np
import pandas as pd
import streamlit as st
from sklearn import covariance
from sklearn import cluster
from sklearn import manifold
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

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
_ = st.session_state.setdefault("edge_model", None)
_ = st.session_state.setdefault("embedding", None)

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

st.markdown("## 1. Retrieve the data from Internet")
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

    st.markdown("## 2. Learning the graph structure")
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
            if st.button("🚀 Fit", key="fit_glasso"):
                with st.spinner("..."):
                    edge_model.fit(X)
                    st.session_state.edge_model = edge_model
        with cols[1]:
            st.markdown("<br/>", unsafe_allow_html=True)
            if st.button("💣 Reset"):
                st.session_state.edge_model = None

if st.session_state.edge_model is not None:
    st.markdown("## 3. Clustering using affinity propagation")
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
    with st.spinner("..."):
        _, labels = cluster.affinity_propagation(
            st.session_state.edge_model.covariance_, random_state=0
        )
    n_labels = labels.max() + 1
    # Show cluster components in a grid
    n_rows = 3
    n_cols = int(np.ceil(n_labels / n_rows))
    i = 0
    for r in range(n_rows):
        cols = st.columns([1] * n_cols)
        for c, col in enumerate(cols):
            if i < n_labels:
                with col:
                    with st.expander(f"Cluster {i + 1}"):
                        for row in st.session_state.names[labels == i]:
                            st.markdown(f"`{row}`")
                        i += 1

    st.markdown("## 4. Embedding in 2D space")
    st.write(
        """
        For visualization purposes, we need to lay out the different symbols on a 2D canvas.
        For this we use [Manifold learning](https://scikit-learn.org/stable/modules/manifold.html#manifold) techniques to retrieve 2D embedding.
        We use a dense eigen_solver to achieve reproducibility (arpack is initiated with the random vectors that we don’t control).
        In addition, we use a large number of neighbors to capture the large-scale structure.
    """
    )
    with st.expander("Embedding"):
        n_neighbors = st.number_input("\# Neighbors", 1, X.shape[0] // 3, 6)
        if st.button("🚀 Fit", key="fit_embedding"):
            with st.spinner("..."):
                node_position_model = manifold.LocallyLinearEmbedding(
                    n_components=2, eigen_solver="dense", n_neighbors=n_neighbors
                )
            embedding = node_position_model.fit_transform(X.T).T
            st.session_state.embedding = embedding

if st.session_state.embedding is not None:
    st.markdown("## 5. Visualization")
    st.write(
        """
        The output of the 3 models are combined in a 2D graph where nodes represents the stocks and edges the:

        - cluster labels are used to define the color of the nodes
        - the sparse covariance model is used to display the strength of the edges
        - the 2D embedding is used to position the nodes in the plan

        This example has a fair amount of visualization-related code, as visualization is crucial here to display the graph.
        One of the challenge is to position the labels minimizing overlap.
        For this we use an heuristic based on the direction of the nearest neighbor along each axis.
    """
    )

    fig = plt.figure(1, facecolor="w", figsize=(10, 8))
    plt.clf()
    ax = plt.axes([0.0, 0.0, 1.0, 1.0])
    plt.axis("off")

    # Plot the graph of partial correlations
    partial_correlations = st.session_state.edge_model.precision_.copy()
    d = 1 / np.sqrt(np.diag(partial_correlations))
    partial_correlations *= d
    partial_correlations *= d[:, np.newaxis]
    non_zero = np.abs(np.triu(partial_correlations, k=1)) > 0.02

    # Plot the nodes using the coordinates of our embedding
    plt.scatter(
        st.session_state.embedding[0],
        st.session_state.embedding[1],
        s=100 * d**2,
        c=labels,
        cmap=plt.cm.nipy_spectral,
    )

    # Plot the edges
    start_idx, end_idx = np.where(non_zero)
    # a sequence of (*line0*, *line1*, *line2*), where::
    #            linen = (x0, y0), (x1, y1), ... (xm, ym)
    segments = [
        [st.session_state.embedding[:, start], st.session_state.embedding[:, stop]]
        for start, stop in zip(start_idx, end_idx)
    ]
    values = np.abs(partial_correlations[non_zero])
    lc = LineCollection(
        segments, zorder=0, cmap=plt.cm.hot_r, norm=plt.Normalize(0, 0.7 * values.max())
    )
    lc.set_array(values)
    lc.set_linewidths(15 * values)
    ax.add_collection(lc)

    # Add a label to each node. The challenge here is that we want to
    # position the labels to avoid overlap with other labels
    for index, (name, label, (x, y)) in enumerate(
        zip(st.session_state.names, labels, st.session_state.embedding.T)
    ):

        dx = x - st.session_state.embedding[0]
        dx[index] = 1
        dy = y - st.session_state.embedding[1]
        dy[index] = 1
        this_dx = dx[np.argmin(np.abs(dy))]
        this_dy = dy[np.argmin(np.abs(dx))]
        if this_dx > 0:
            horizontalalignment = "left"
            x = x + 0.002
        else:
            horizontalalignment = "right"
            x = x - 0.002
        if this_dy > 0:
            verticalalignment = "bottom"
            y = y + 0.002
        else:
            verticalalignment = "top"
            y = y - 0.002
        plt.text(
            x,
            y,
            name,
            size=10,
            horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment,
            bbox=dict(
                facecolor="w",
                edgecolor=plt.cm.nipy_spectral(label / float(n_labels)),
                alpha=0.6,
            ),
        )

    plt.xlim(
        st.session_state.embedding[0].min()
        - 0.15 * st.session_state.embedding[0].ptp(),
        st.session_state.embedding[0].max()
        + 0.10 * st.session_state.embedding[0].ptp(),
    )
    plt.ylim(
        st.session_state.embedding[1].min()
        - 0.03 * st.session_state.embedding[1].ptp(),
        st.session_state.embedding[1].max()
        + 0.03 * st.session_state.embedding[1].ptp(),
    )

    st.pyplot(fig)
