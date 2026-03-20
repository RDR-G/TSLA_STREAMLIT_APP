import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pickle
import sqlite3
import tempfile
import os
import re
import io
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="StockVision – Tesla Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background-color: #0d1117; }
    .stApp { background-color: #0d1117; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
        border-right: 1px solid #30363d;
    }
    section[data-testid="stSidebar"] * { color: #e6edf3 !important; }

    /* ── Compact sidebar spacing (font sizes unchanged) ── */
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 0.75rem !important;
        padding-bottom: 0.5rem !important;
    }
    section[data-testid="stSidebar"] h2 {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }
    section[data-testid="stSidebar"] h3 {
        margin-top: 4px !important;
        margin-bottom: 2px !important;
    }
    section[data-testid="stSidebar"] hr {
        margin: 6px 0 !important;
        border-color: #30363d !important;
    }
    section[data-testid="stSidebar"] .success-box,
    section[data-testid="stSidebar"] .warn-box {
        padding: 6px 12px !important;
        margin: 3px 0 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div {
        gap: 0.3rem !important;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploader"] > div {
        padding: 8px !important;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label {
        padding-top: 2px !important;
        padding-bottom: 2px !important;
    }
    section[data-testid="stSidebar"] .stButton button {
        padding: 4px 12px !important;
    }

    div[data-testid="metric-container"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 16px;
    }
    div[data-testid="metric-container"] label { color: #8b949e !important; font-size: 13px !important; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #e6edf3 !important; font-size: 24px !important; font-weight: 700 !important;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricDelta"] { font-size: 13px !important; }

    h1, h2, h3 { color: #e6edf3 !important; }
    h1 { font-size: 2rem !important; font-weight: 700 !important; }

    .section-header {
        background: linear-gradient(90deg, #e8273b 0%, #c0392b 100%);
        color: white !important;
        padding: 10px 20px;
        border-radius: 8px;
        font-size: 18px;
        font-weight: 700;
        margin: 20px 0 10px 0;
    }
    .info-box {
        background: #161b22;
        border-left: 4px solid #e8273b;
        border-radius: 6px;
        padding: 14px 18px;
        margin: 10px 0;
        color: #c9d1d9;
    }
    .insight-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 6px 0;
        color: #c9d1d9;
    }
    .insight-box strong { color: #e8273b; }
    p, li, span { color: #c9d1d9 !important; }
    .stDataFrame { border-radius: 10px; overflow: hidden; }
    div[data-testid="stTabs"] button { color: #8b949e !important; }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #e8273b !important;
        border-bottom: 2px solid #e8273b !important;
    }
    .success-box {
        background: #0d2818;
        border-left: 4px solid #3fb950;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 6px 0;
        color: #c9d1d9;
        font-size: 13px;
    }
    .warn-box {
        background: #2b1d0e;
        border-left: 4px solid #f0a500;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 6px 0;
        color: #c9d1d9;
        font-size: 13px;
    }
    .source-badge {
        display: inline-block;
        background: #e8273b;
        color: white !important;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        margin-left: 8px;
    }
</style>
""", unsafe_allow_html=True)

REQUIRED_COLS   = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
MODEL_FEAT_COLS = ['Close', 'High', 'Low', 'Open', 'Volume',
                   'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3', 'Close_Lag_4', 'Close_Lag_5',
                   'Rolling_Mean_7', 'Rolling_Mean_30', 'Rolling_Std_7', 'Rolling_Std_30']

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0d1117",
    plot_bgcolor="#0d1117",
    font=dict(color="#c9d1d9"),
    xaxis=dict(gridcolor="#21262d", showline=True, linecolor="#30363d"),
    yaxis=dict(gridcolor="#21262d", showline=True, linecolor="#30363d"),
    margin=dict(l=40, r=20, t=50, b=40),
)

def validate_and_prepare(df: pd.DataFrame):
    """
    Validate that all required columns are present (case-insensitive),
    keep only those columns, coerce types, and set Date as index.
    Returns (prepared_df, error_message_or_None).
    """
    
    col_map = {c.strip().lower(): c for c in df.columns}
    
    rename_map = {}
    missing    = []
    for req in REQUIRED_COLS:
        if req in df.columns:
            continue                          #exact match
        elif req.lower() in col_map:
            rename_map[col_map[req.lower()]] = req   #case mismatch
        else:
            missing.append(req)

    if missing:
        return None, (
            f"Missing required column(s): **{', '.join(missing)}**\n\n"
            f"Required columns: `{', '.join(REQUIRED_COLS)}`"
        )

    if rename_map:
        df = df.rename(columns=rename_map)
    df = df[REQUIRED_COLS].copy()

    try:
        df['Date'] = pd.to_datetime(df['Date'])
    except Exception as e:
        return None, f"Could not parse **Date** column: {e}"

    df = df.sort_values('Date').reset_index(drop=True)
    df.set_index('Date', inplace=True)

    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    before = len(df)
    df     = df.dropna()
    after  = len(df)
    dropped_msg = f" ({before - after} rows with NaN values dropped)" if before != after else ""

    if len(df) < 60:
        return None, "Dataset too small — need at least 60 rows after cleaning."

    return df, dropped_msg or None  

@st.cache_data(show_spinner=False)
def _load_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def _load_json(file_bytes: bytes) -> pd.DataFrame:
    for orient in ['records', 'columns', 'index', 'split', 'values', None]:
        try:
            kwargs = {"orient": orient} if orient else {}
            df = pd.read_json(io.BytesIO(file_bytes), **kwargs)
            if len(df.columns) >= 5:
                return df
        except Exception:
            continue
    raise ValueError("Could not parse JSON file with any standard orientation.")


@st.cache_data(show_spinner=False)
def _load_sql(file_bytes: bytes, filename: str, table_name: str) -> pd.DataFrame:
    ext = os.path.splitext(filename)[-1].lower()

    if ext in ('.db', '.sqlite', '.sqlite3'):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            conn = sqlite3.connect(tmp_path)
            df   = pd.read_sql(f'SELECT * FROM "{table_name}"', conn)
            conn.close()
        finally:
            os.unlink(tmp_path)

    else: 
        sql_text = file_bytes.decode('utf-8', errors='replace')
        conn     = sqlite3.connect(':memory:')
        conn.executescript(sql_text)
        df = pd.read_sql(f'SELECT * FROM "{table_name}"', conn)
        conn.close()

    return df


@st.cache_data(show_spinner=False)
def _get_sql_tables(file_bytes: bytes, filename: str) -> list:
    ext = os.path.splitext(filename)[-1].lower()

    if ext in ('.db', '.sqlite', '.sqlite3'):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            conn   = sqlite3.connect(tmp_path)
            tables = pd.read_sql(
                "SELECT name FROM sqlite_master WHERE type='table'", conn
            )['name'].tolist()
            conn.close()
        finally:
            os.unlink(tmp_path)

    else:
        sql_text = file_bytes.decode('utf-8', errors='replace')
        conn     = sqlite3.connect(':memory:')
        conn.executescript(sql_text)
        tables = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table'", conn
        )['name'].tolist()
        conn.close()

    return tables


@st.cache_data(show_spinner=False)
def _load_drive(share_url: str, file_format: str) -> bytes:
    """Download a publicly shared Google Drive file and return raw bytes."""
    import urllib.request, urllib.error

    patterns = [
        r'/file/d/([a-zA-Z0-9_-]+)',
        r'[?&]id=([a-zA-Z0-9_-]+)',
        r'/open\?id=([a-zA-Z0-9_-]+)',
        r'/uc\?.*id=([a-zA-Z0-9_-]+)',
    ]
    file_id = None
    for p in patterns:
        m = re.search(p, share_url)
        if m:
            file_id = m.group(1)
            break

    if not file_id:
        raise ValueError("Could not extract file ID from the Google Drive URL.")

    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

    req = urllib.request.Request(
        download_url,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read()
    except urllib.error.URLError as e:
        raise ConnectionError(f"Failed to download file: {e}")

    if content[:100].lstrip().startswith(b'<!'):
        token_m = re.search(rb'confirm=([0-9A-Za-z_-]+)', content)
        if token_m:
            confirm = token_m.group(1).decode()
            dl_url2 = f"{download_url}&confirm={confirm}"
            with urllib.request.urlopen(
                urllib.request.Request(dl_url2, headers={"User-Agent": "Mozilla/5.0"}),
                timeout=30
            ) as resp:
                content = resp.read()
        else:
            raise ValueError(
                "Google Drive returned an HTML page instead of the file. "
                "Make sure the file is shared with 'Anyone with the link'."
            )
    return content

@st.cache_data(show_spinner=False)
def engineer_features(_df: pd.DataFrame) -> pd.DataFrame:
    data = _df.copy()
    for i in range(1, 6):
        data[f'Close_Lag_{i}'] = data['Close'].shift(i)
    data['Rolling_Mean_7']  = data['Close'].rolling(7).mean()
    data['Rolling_Mean_30'] = data['Close'].rolling(30).mean()
    data['Rolling_Std_7']   = data['Close'].rolling(7).std()
    data['Rolling_Std_30']  = data['Close'].rolling(30).std()
    data['Target']          = data['Close'].shift(-1)
    return data.dropna()


@st.cache_data(show_spinner=False)
def evaluate_model(_model_data: pd.DataFrame, model_bytes: bytes):
    """
    Evaluate the pre-trained pickled model on an 80/20 time-ordered split.
    Returns the same tuple signature as the old train_linear_regression().
    """
    model        = pickle.loads(model_bytes)
    feature_cols = list(model.feature_names_in_)  

    X = _model_data[feature_cols]
    y = _model_data['Target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    preds   = model.predict(X_test)
    metrics = {
        'MAE':  mean_absolute_error(y_test, preds),
        'RMSE': np.sqrt(mean_squared_error(y_test, preds)),
        'R2':   r2_score(y_test, preds),
    }
    return model, X_train, X_test, y_train, y_test, preds, metrics, feature_cols

def risk_metrics(df):
    returns  = df['Close'].pct_change().dropna()
    rf_daily = 0.02 / 252
    excess   = returns - rf_daily
    sharpe   = excess.mean() / excess.std() * np.sqrt(252)
    roll_max = df['Close'].cummax()
    drawdown = (df['Close'] - roll_max) / roll_max
    max_dd   = drawdown.min()
    ann_vol  = returns.std() * np.sqrt(252)
    ann_ret  = returns.mean() * 252
    total_ret = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
    return {
        'Total Return (%)':          round(total_ret, 2),
        'Annualized Return (%)':     round(ann_ret * 100, 2),
        'Annualized Volatility (%)': round(ann_vol * 100, 2),
        'Sharpe Ratio':              round(sharpe, 3),
        'Max Drawdown (%)':          round(max_dd * 100, 2),
        'Best Day (%)':              round(returns.max() * 100, 2),
        'Worst Day (%)':             round(returns.min() * 100, 2),
        'Positive Days (%)':         round((returns > 0).mean() * 100, 1),
    }

with st.sidebar:
    st.markdown("## StockVision")
    st.markdown("**Tesla (TSLA) Analysis**")
    st.markdown("---")

    st.markdown("### Prediction Model")
    _MODEL_PATH = os.path.join(os.path.dirname(__file__), "TSLA_linear_model.pkl")
    model_bytes = None
    try:
        with open(_MODEL_PATH, "rb") as _f:
            model_bytes = _f.read()
        _test_model = pickle.loads(model_bytes)
        st.markdown(
            "<div class='success-box'>✔ Linear Regression model loaded — "
            f"{_test_model.n_features_in_} features</div>",
            unsafe_allow_html=True
        )
    except FileNotFoundError:
        st.markdown(
            "<div class='warn-box'>⚠ <strong>TSLA_linear_model.pkl</strong> not found. "
            "Place the file in the same folder as this app.</div>",
            unsafe_allow_html=True
        )
    except Exception as _e:
        st.markdown(
            f"<div class='warn-box'>⚠ Could not load model: {_e}</div>",
            unsafe_allow_html=True
        )
        model_bytes = None

    st.markdown("---")

    st.markdown("### Data Source")
    source = st.radio(
        "Select data source",
        ["CSV", "JSON", "SQL Database", "Google Drive"],
        label_visibility="collapsed"
    )

    df          = None
    source_label = ""
    load_error   = None
    info_msg     = None

    if source == "CSV":
        source_label = "CSV"
        csv_file = st.file_uploader(
            "Upload CSV file",
            type=["csv"],
            key="csv_upload"
        )
        if csv_file:
            try:
                raw_df = _load_csv(csv_file.getvalue())
                df, msg = validate_and_prepare(raw_df)
                if df is None:
                    load_error = msg
                else:
                    info_msg = msg
            except Exception as e:
                load_error = f"Failed to read CSV: {e}"

    elif source == "JSON":
        source_label = "JSON"
        json_file = st.file_uploader(
            "Upload JSON file",
            type=["json"],
            key="json_upload"
        )
        if json_file:
            try:
                raw_df = _load_json(json_file.getvalue())
                df, msg = validate_and_prepare(raw_df)
                if df is None:
                    load_error = msg
                else:
                    info_msg = msg
            except Exception as e:
                load_error = f"Failed to read JSON: {e}"

    elif source == "SQL Database":
        source_label = "SQL"
        sql_file = st.file_uploader(
            "Upload SQLite database or SQL dump",
            type=["db", "sqlite", "sqlite3", "sql"],
            key="sql_upload",
            help="Supported: .db / .sqlite / .sqlite3 (SQLite database) or .sql (SQL dump text file)"
        )
        if sql_file:
            try:
                sql_bytes  = sql_file.getvalue()
                sql_tables = _get_sql_tables(sql_bytes, sql_file.name)

                if not sql_tables:
                    load_error = "No tables found in the SQL file."
                else:
                    selected_table = st.selectbox(
                        "Select table",
                        sql_tables,
                        key="sql_table"
                    )
                    if selected_table:
                        raw_df = _load_sql(sql_bytes, sql_file.name, selected_table)
                        df, msg = validate_and_prepare(raw_df)
                        if df is None:
                            load_error = msg
                        else:
                            info_msg = msg
            except Exception as e:
                load_error = f"Failed to read SQL file: {e}"

    elif source == "Google Drive":
        source_label = "Drive"
        drive_url = st.text_input(
            "Paste Google Drive share link",
            placeholder="https://drive.google.com/file/d/.../view?usp=sharing",
            key="drive_url"
        )
        drive_fmt = st.selectbox(
            "File format",
            ["CSV", "JSON"],
            key="drive_fmt",
            help="Select the format of the file in Drive"
        )
        load_drive_btn = st.button("Load from Drive", key="drive_btn")

        if load_drive_btn and drive_url.strip():
            with st.spinner("Downloading from Google Drive…"):
                try:
                    content = _load_drive(drive_url.strip(), drive_fmt)
                    if drive_fmt == "CSV":
                        raw_df = pd.read_csv(io.BytesIO(content))
                    else:
                        raw_df = pd.read_json(io.BytesIO(content))
                    df, msg = validate_and_prepare(raw_df)
                    if df is None:
                        load_error = msg
                    else:
                        info_msg = msg
                    st.session_state['drive_df']  = df
                    st.session_state['drive_err'] = load_error
                    st.session_state['drive_info'] = info_msg
                except Exception as e:
                    load_error = f"Drive download failed: {e}"
                    st.session_state['drive_err'] = load_error
                    st.session_state['drive_df']  = None

        if df is None and 'drive_df' in st.session_state:
            df         = st.session_state.get('drive_df')
            load_error = st.session_state.get('drive_err')
            info_msg   = st.session_state.get('drive_info')

        if not drive_url.strip():
            st.caption("Make sure the file is shared as **Anyone with the link**.")

    if load_error:
        st.markdown(f"<div class='warn-box'>{load_error}</div>", unsafe_allow_html=True)
    elif df is not None:
        st.markdown(
            f"<div class='success-box'>{len(df):,} rows loaded "
            f"<span class='source-badge'>{source_label}</span></div>",
            unsafe_allow_html=True
        )
        if info_msg:
            st.markdown(f"<div class='warn-box'>{info_msg}</div>", unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio("**Navigate**", [" Dashboard", " EDA", " Prediction"])
    st.markdown("---")
    st.markdown("**Model:** **Linear Regression**")
    st.caption("StockVision · Built with Streamlit")


if df is None:
    st.markdown("---")
    st.info("Please upload your data using the sidebar to get started.")
    st.markdown("""
    <div class='info-box'>
    <b>Supported data sources:</b><br>
    &nbsp;&nbsp;• <b>CSV</b> — upload a <code>.csv</code> file<br>
    &nbsp;&nbsp;• <b>JSON</b> — upload a <code>.json</code> file (records or columns orientation)<br>
    &nbsp;&nbsp;• <b>SQL Database</b> — upload a <code>.db</code> / <code>.sqlite</code> / <code>.sql</code> file<br>
    &nbsp;&nbsp;• <b>Google Drive</b> — paste a shareable link (Anyone with the link)<br><br>
    <b>Required columns in your dataset:</b><br>
    <code>Date, Open, High, Low, Close, Volume</code>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


if page == " Dashboard":
    st.markdown("<h1>StockVision — Tesla Stock Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p>Comprehensive analysis of Tesla Inc. (TSLA) — June 2010 to December 2025</p>", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Key Metrics</div>", unsafe_allow_html=True)
    rm = risk_metrics(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest Close",  f"${df['Close'].iloc[-1]:.2f}",
              delta=f"{((df['Close'].iloc[-1]/df['Close'].iloc[-2])-1)*100:.2f}%")
    c2.metric("Total Return",  f"{rm['Total Return (%)']:,.1f}%")
    c3.metric("Sharpe Ratio",  f"{rm['Sharpe Ratio']}")
    c4.metric("Max Drawdown",  f"{rm['Max Drawdown (%)']:.2f}%")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Ann. Return",    f"{rm['Annualized Return (%)']:.2f}%")
    c6.metric("Ann. Volatility",f"{rm['Annualized Volatility (%)']:.2f}%")
    c7.metric("Best Day",       f"{rm['Best Day (%)']:.2f}%")
    c8.metric("Positive Days",  f"{rm['Positive Days (%)']:.1f}%")

    st.markdown("<div class='section-header'>Price History</div>", unsafe_allow_html=True)
    date_range = st.select_slider(
        "Select date range",
        options=["1Y", "3Y", "5Y", "10Y", "All"],
        value="All"
    )
    cutoffs = {"1Y": -252, "3Y": -252*3, "5Y": -252*5, "10Y": -252*10, "All": 0}
    idx = cutoffs[date_range]
    df_plot = df.iloc[idx:] if idx != 0 else df

    df_plot = df_plot.copy()
    df_plot['MA20']  = df_plot['Close'].rolling(20).mean()
    df_plot['MA50']  = df_plot['Close'].rolling(50).mean()
    df_plot['MA200'] = df_plot['Close'].rolling(200).mean()

    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Close'],
        name='Close', line=dict(color='#e8273b', width=1.5)))
    fig_price.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA20'],
        name='MA 20', line=dict(color='#f0a500', width=1, dash='dot')))
    fig_price.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA50'],
        name='MA 50', line=dict(color='#1f9eff', width=1, dash='dot')))
    fig_price.add_trace(go.Scatter(x=df_plot.index, y=df_plot['MA200'],
        name='MA 200', line=dict(color='#58a6ff', width=1.2)))
    fig_price.update_layout(**PLOTLY_LAYOUT, title='Tesla Closing Price with Moving Averages',
                             xaxis_title='Date', yaxis_title='Price ($)', height=450,
                             legend=dict(bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig_price, use_container_width=True)

    st.markdown("<div class='section-header'>Candlestick Chart (Last 3 Months)</div>", unsafe_allow_html=True)
    df_candle = df.iloc[-63:].copy()
    fig_candle = go.Figure(go.Candlestick(
        x=df_candle.index,
        open=df_candle['Open'], high=df_candle['High'],
        low=df_candle['Low'],   close=df_candle['Close'],
        increasing_line_color='#3fb950', decreasing_line_color='#f85149'
    ))
    fig_candle.update_layout(**PLOTLY_LAYOUT, title='Candlestick — Recent 3 Months',
                              xaxis_title='Date', yaxis_title='Price ($)', height=400,
                              xaxis_rangeslider_visible=False)
    st.plotly_chart(fig_candle, use_container_width=True)

    st.markdown("<div class='section-header'>Key Insights</div>", unsafe_allow_html=True)
    insights = [
        ("Total Return",   f"Tesla delivered a remarkable <strong>{rm['Total Return (%)']:,.1f}%</strong> total return across the dataset period."),
        ("Volatility",     f"Annualized volatility of <strong>{rm['Annualized Volatility (%)']:.1f}%</strong> confirms Tesla as a high-risk, high-reward growth stock."),
        ("Sharpe Ratio",   f"A Sharpe ratio of <strong>{rm['Sharpe Ratio']}</strong> indicates returns reasonably compensate for risk taken."),
        ("Drawdown",       f"Maximum drawdown of <strong>{rm['Max Drawdown (%)']:.1f}%</strong> highlights severe downside risk during correction periods."),
        ("Positive Days",  f"Tesla closes up on <strong>{rm['Positive Days (%)']:.1f}%</strong> of trading days."),
    ]
    for title, body in insights:
        st.markdown(f"<div class='insight-box'><strong>{title}:</strong> {body}</div>", unsafe_allow_html=True)

elif page == " EDA":
    st.markdown("<h1>Exploratory Data Analysis</h1>", unsafe_allow_html=True)
    st.markdown("<p>Deep dive into Tesla stock data — distributions, correlations, volatility and time series patterns.</p>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Data Overview", "Price Analysis", "Volatility", "Correlations", "Time Series"
    ])

    with tab1:
        st.markdown("<div class='section-header'>Dataset Overview</div>", unsafe_allow_html=True)
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Rows",     f"{len(df):,}")
        col2.metric("Columns",        f"{len(df.columns)}")
        col3.metric("Start Date",     f"{df.index.min().date()}")
        col4.metric("End Date",       f"{df.index.max().date()}")
        col5.metric("Missing Values", f"{df.isnull().sum().sum()}")

        st.markdown("**First 10 Rows**")
        st.dataframe(df.head(10), use_container_width=True)

        st.markdown("**Descriptive Statistics**")
        st.dataframe(df.describe().round(4), use_container_width=True)

        st.markdown("**Data Types & Missing Values**")
        dtype_df = pd.DataFrame({
            'Column':  df.columns,
            'dtype':   df.dtypes.values,
            'Missing': df.isnull().sum().values,
            'Unique':  df.nunique().values,
        })
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)

    with tab2:
        st.markdown("<div class='section-header'>Price Distribution Analysis</div>", unsafe_allow_html=True)
        price_col = st.selectbox("Select column", ['Close', 'Open', 'High', 'Low'])

        col_a, col_b = st.columns(2)
        with col_a:
            fig_hist = px.histogram(df, x=price_col, nbins=80,
                title=f'{price_col} Price Distribution',
                color_discrete_sequence=['#e8273b'])
            fig_hist.update_layout(**PLOTLY_LAYOUT, height=350)
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_b:
            fig_box = go.Figure()
            for col_name, color in zip(['Open', 'High', 'Low', 'Close'],
                                        ['#58a6ff','#3fb950','#f0a500','#e8273b']):
                fig_box.add_trace(go.Box(y=df[col_name], name=col_name,
                    marker_color=color, boxmean=True))
            fig_box.update_layout(**PLOTLY_LAYOUT, title='OHLC Price Box Plots', height=350)
            st.plotly_chart(fig_box, use_container_width=True)

        st.markdown("<div class='section-header'>Yearly Average Close Price</div>", unsafe_allow_html=True)
        yearly = df['Close'].resample('YE').mean().reset_index()
        yearly.columns = ['Year', 'Avg Close']
        yearly['Year'] = yearly['Year'].dt.year
        fig_yr = px.bar(yearly, x='Year', y='Avg Close',
            color='Avg Close', color_continuous_scale='Reds',
            title='Average Closing Price by Year')
        fig_yr.update_layout(**PLOTLY_LAYOUT, height=380)
        st.plotly_chart(fig_yr, use_container_width=True)

    
    with tab3:
        st.markdown("<div class='section-header'>Rolling Volatility</div>", unsafe_allow_html=True)
        returns       = df['Close'].pct_change().dropna()
        roll_vol_30   = returns.rolling(30).std()  * np.sqrt(252) * 100
        roll_vol_90   = returns.rolling(90).std()  * np.sqrt(252) * 100
        roll_vol_252  = returns.rolling(252).std() * np.sqrt(252) * 100

        fig_rvol = go.Figure()
        fig_rvol.add_trace(go.Scatter(x=roll_vol_30.index,  y=roll_vol_30,  name='30-day',  line=dict(color='#f0a500', width=1)))
        fig_rvol.add_trace(go.Scatter(x=roll_vol_90.index,  y=roll_vol_90,  name='90-day',  line=dict(color='#1f9eff', width=1.5)))
        fig_rvol.add_trace(go.Scatter(x=roll_vol_252.index, y=roll_vol_252, name='252-day', line=dict(color='#e8273b', width=2)))
        fig_rvol.update_layout(**PLOTLY_LAYOUT, title='Annualized Rolling Volatility (%)',
                                xaxis_title='Date', yaxis_title='Volatility (%)', height=420)
        st.plotly_chart(fig_rvol, use_container_width=True)

        st.markdown("<div class='section-header'>Drawdown Analysis</div>", unsafe_allow_html=True)
        roll_max = df['Close'].cummax()
        drawdown = (df['Close'] - roll_max) / roll_max * 100
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=drawdown.index, y=drawdown,
            fill='tozeroy', fillcolor='rgba(232,39,59,0.2)',
            line=dict(color='#e8273b', width=1.5), name='Drawdown'))
        fig_dd.update_layout(**PLOTLY_LAYOUT, title='Drawdown from All-Time High (%)',
                              xaxis_title='Date', yaxis_title='Drawdown (%)', height=380)
        st.plotly_chart(fig_dd, use_container_width=True)

        st.markdown("<div class='section-header'>Rolling Sharpe Ratio (252-day)</div>", unsafe_allow_html=True)
        rf_daily = 0.02 / 252
        excess   = returns - rf_daily
        rolling_sharpe  = (excess.rolling(252).mean() / excess.rolling(252).std()) * np.sqrt(252)
        overall_sharpe  = (excess.mean() / excess.std()) * np.sqrt(252)
        fig_sharpe = go.Figure()
        fig_sharpe.add_trace(go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe,
            name='Rolling Sharpe', line=dict(color='#3fb950', width=1.5)))
        fig_sharpe.add_hline(y=overall_sharpe, line_dash='dash', line_color='#f0a500',
            annotation_text=f'Overall: {overall_sharpe:.3f}')
        fig_sharpe.add_hline(y=0, line_color='#8b949e', line_width=1)
        fig_sharpe.update_layout(**PLOTLY_LAYOUT, title='Rolling Sharpe Ratio',
                                  xaxis_title='Date', yaxis_title='Sharpe Ratio', height=380)
        st.plotly_chart(fig_sharpe, use_container_width=True)

    with tab4:
        st.markdown("<div class='section-header'>Correlation Matrix</div>", unsafe_allow_html=True)
        corr = df.corr()
        fig_corr = px.imshow(corr, text_auto='.3f', color_continuous_scale='RdBu_r',
            title='Feature Correlation Matrix', aspect='auto')
        fig_corr.update_layout(**PLOTLY_LAYOUT, height=480)
        st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown("<div class='section-header'>Volume vs Close Price</div>", unsafe_allow_html=True)
        fig_scatter = px.scatter(df.reset_index(), x='Volume', y='Close',
            color='Close', color_continuous_scale='Reds',
            title='Volume vs. Close Price', opacity=0.4)
        fig_scatter.update_layout(**PLOTLY_LAYOUT, height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("<div class='section-header'>Feature Scatter</div>", unsafe_allow_html=True)
        c_x, c_y = st.columns(2)
        feat_x = c_x.selectbox("X-axis", df.columns.tolist(), index=0)
        feat_y = c_y.selectbox("Y-axis", df.columns.tolist(), index=1)
        fig_fs = px.scatter(df.reset_index(), x=feat_x, y=feat_y,
            opacity=0.4, color_discrete_sequence=['#58a6ff'],
            title=f'{feat_x} vs {feat_y}', trendline='ols')
        fig_fs.update_layout(**PLOTLY_LAYOUT, height=400)
        st.plotly_chart(fig_fs, use_container_width=True)

    with tab5:
        st.markdown("<div class='section-header'>Trend & Seasonality</div>", unsafe_allow_html=True)
        close      = df['Close'].copy()
        trend_365  = close.rolling(365, center=True).mean()
        detrended  = close / trend_365
        seasonal   = detrended.groupby(detrended.index.dayofyear).transform('mean')
        residual   = close / (trend_365 * seasonal)

        fig_decomp = make_subplots(rows=4, cols=1, shared_xaxes=True,
            subplot_titles=('Original', 'Trend (365-day MA)', 'Seasonal', 'Residual'))
        for row, (series, color, name) in enumerate([
            (close,     '#e8273b', 'Close'),
            (trend_365, '#58a6ff', 'Trend'),
            (seasonal,  '#3fb950', 'Seasonal'),
            (residual,  '#f0a500', 'Residual'),
        ], 1):
            fig_decomp.add_trace(go.Scatter(x=series.index, y=series, name=name,
                line=dict(color=color, width=1.2)), row=row, col=1)
        fig_decomp.update_layout(**PLOTLY_LAYOUT, height=700, showlegend=True,
                                  title='Seasonal Decomposition (Multiplicative)')
        st.plotly_chart(fig_decomp, use_container_width=True)

        st.markdown("<div class='section-header'>Returns Autocorrelation (Lags 1–30)</div>", unsafe_allow_html=True)
        daily_returns = df['Close'].pct_change().dropna()
        lags     = range(1, 31)
        acf_vals = [daily_returns.autocorr(lag=l) for l in lags]
        fig_acf  = go.Figure()
        fig_acf.add_trace(go.Bar(x=list(lags), y=acf_vals,
            marker_color=['#3fb950' if v > 0 else '#e8273b' for v in acf_vals]))
        fig_acf.add_hline(y= 1.96/np.sqrt(len(daily_returns)), line_dash='dash', line_color='#f0a500')
        fig_acf.add_hline(y=-1.96/np.sqrt(len(daily_returns)), line_dash='dash', line_color='#f0a500')
        fig_acf.update_layout(**PLOTLY_LAYOUT, title='Autocorrelation of Daily Returns',
                               xaxis_title='Lag', yaxis_title='Autocorrelation', height=360)
        st.plotly_chart(fig_acf, use_container_width=True)

elif page == " Prediction":
    st.markdown("<h1>Linear Regression — Price Prediction</h1>", unsafe_allow_html=True)
    st.markdown("<p>Evaluate the pre-trained Linear Regression model and explore next-day price forecasts.</p>", unsafe_allow_html=True)

    if model_bytes is None:
        st.warning(
            "**No model loaded.** Make sure `TSLA_linear_model.pkl` is placed "
            "in the same directory as this app and restart."
        )
        st.stop()

    loaded_model = pickle.loads(model_bytes)
    required_feats = list(loaded_model.feature_names_in_)

    base_needed = [f for f in ['Open', 'High', 'Low', 'Close', 'Volume']
                   if f in required_feats]
    missing_base = [f for f in base_needed if f not in df.columns]
    if missing_base:
        st.error(f"The loaded dataset is missing columns required by the model: `{missing_base}`")
        st.stop()

    with st.spinner("Engineering features and evaluating model…"):
        model_data = engineer_features(df)

        missing_feats = [f for f in required_feats if f not in model_data.columns]
        if missing_feats:
            st.error(f"Could not compute model features: `{missing_feats}`")
            st.stop()

        model, X_train, X_test, y_train, y_test, preds, metrics, feature_cols = \
            evaluate_model(model_data, model_bytes)

    st.markdown(f"""
    <div class='success-box'>
    <strong>Pre-trained model loaded from .pkl</strong> &nbsp;|&nbsp;
    Features: <strong>{model.n_features_in_}</strong> &nbsp;|&nbsp;
    Algorithm: <strong>Linear Regression</strong>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Model Performance</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE",       f"${metrics['MAE']:.2f}",  help="Mean Absolute Error")
    c2.metric("RMSE",      f"${metrics['RMSE']:.2f}", help="Root Mean Squared Error")
    c3.metric("R² Score",  f"{metrics['R2']:.4f}",    help="Variance explained by model")

    st.markdown(f"""
    <div class='info-box'>
    <strong>Model Interpretation:</strong> R² of <strong>{metrics['R2']:.4f}</strong> means the model explains
    <strong>{metrics['R2']*100:.1f}%</strong> of variance in next-day close price.
    The average prediction error is <strong>${metrics['MAE']:.2f}</strong> per day.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Actual vs Predicted</div>", unsafe_allow_html=True)
    fig_avp = go.Figure()
    fig_avp.add_trace(go.Scatter(x=y_test.index, y=y_test.values,
        name='Actual', line=dict(color='#58a6ff', width=1.8)))
    fig_avp.add_trace(go.Scatter(x=y_test.index, y=preds,
        name='Predicted', line=dict(color='#e8273b', width=1.5, dash='dot')))
    fig_avp.update_layout(**PLOTLY_LAYOUT, title='Linear Regression: Actual vs Predicted Close Price',
                           xaxis_title='Date', yaxis_title='Price ($)', height=450,
                           legend=dict(bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig_avp, use_container_width=True)

    st.markdown("<div class='section-header'>Next-Day Price Prediction</div>", unsafe_allow_html=True)
    st.markdown("Adjust the last known values to simulate a custom next-day forecast.")

    last_row = model_data[feature_cols].iloc[-1].copy()

    col_a, col_b, col_c = st.columns(3)
    last_close = float(df['Close'].iloc[-1])
    new_close  = col_a.number_input("Latest Close ($)", value=round(last_close, 2), step=1.0)
    new_high   = col_b.number_input("Latest High ($)",  value=round(float(df['High'].iloc[-1]), 2), step=1.0)
    new_low    = col_c.number_input("Latest Low ($)",   value=round(float(df['Low'].iloc[-1]), 2), step=1.0)

    input_row = last_row.copy()
    for feat in feature_cols:
        if feat in ('Close', 'Close_Lag_1'):
            input_row[feat] = new_close
        elif feat == 'High':
            input_row[feat] = new_high
        elif feat == 'Low':
            input_row[feat] = new_low

    next_pred  = model.predict(input_row.values.reshape(1, -1))[0]
    change     = next_pred - new_close
    pct_change = (change / new_close) * 100

    st.markdown("---")
    p1, p2, p3 = st.columns(3)
    p1.metric("Current Close",            f"${new_close:.2f}")
    p2.metric("Predicted Next-Day Close",  f"${next_pred:.2f}",
              delta=f"{change:+.2f} ({pct_change:+.2f}%)")
    p3.metric("Model R²",                  f"{metrics['R2']:.4f}")

    st.markdown(f"""
    <div class='info-box'>
    Based on the pre-trained Linear Regression model with lag features and rolling statistics,
    the predicted next closing price is <strong>${next_pred:.2f}</strong> —
    a change of <strong>{change:+.2f} ({pct_change:+.2f}%)</strong>
    from today's close of <strong>${new_close:.2f}</strong>.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Train / Test Split</div>", unsafe_allow_html=True)
    fig_split = go.Figure()
    fig_split.add_trace(go.Scatter(x=X_train.index, y=y_train.values,
        name='Training Set', line=dict(color='#58a6ff', width=1.5)))
    fig_split.add_trace(go.Scatter(x=X_test.index, y=y_test.values,
        name='Test Set', line=dict(color='#3fb950', width=1.5)))
    fig_split.update_layout(**PLOTLY_LAYOUT, title='Train / Test Split',
                             xaxis_title='Date', yaxis_title='Close Price ($)',
                             height=380, legend=dict(bgcolor='rgba(0,0,0,0)'))
    st.plotly_chart(fig_split, use_container_width=True)

    col_info1, col_info2 = st.columns(2)
    col_info1.markdown(
        f"<div class='insight-box'><strong>Training samples:</strong> {len(X_train):,} rows (80%)</div>",
        unsafe_allow_html=True
    )
    col_info2.markdown(
        f"<div class='insight-box'><strong>Test samples:</strong> {len(X_test):,} rows (20%)</div>",
        unsafe_allow_html=True
    )
