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
import datetime

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="StockVision – Stock Analysis",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    /* ── Global scale fix for Streamlit Cloud ── */
    html { font-size: 14px; }
    .block-container {
        max-width: 1200px !important;
        padding-top: 1rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    .main { background-color: #0d1117; }
    .stApp { background-color: #0d1117; }

    /* Hide sidebar entirely */
    section[data-testid="stSidebar"] { display: none !important; }
    button[data-testid="collapsedControl"] { display: none !important; }

    /* Top-level tab nav styling */
    div[data-testid="stTabs"] > div:first-child {
        border-bottom: 2px solid #21262d;
        margin-bottom: 12px;
    }
    div[data-testid="stTabs"] button {
        color: #8b949e !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        padding: 8px 18px !important;
        border-radius: 0 !important;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #e8273b !important;
        border-bottom: 3px solid #e8273b !important;
        font-weight: 700 !important;
    }
    div[data-testid="stTabs"] button:hover {
        color: #e6edf3 !important;
        background: rgba(232,39,59,0.08) !important;
    }

    /* Expander styling */
    div[data-testid="stExpander"] {
        background: #161b22;
        border: 1px solid #30363d !important;
        border-radius: 10px !important;
        margin-bottom: 16px;
    }
    div[data-testid="stExpander"] summary {
        color: #e6edf3 !important;
        font-weight: 600;
        font-size: 14px !important;
    }

    /* Metric containers — broad selectors to cover all Streamlit versions */
    div[data-testid="metric-container"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 14px !important;
    }
    div[data-testid="metric-container"] label,
    div[data-testid="stMetricLabel"],
    div[data-testid="stMetricLabel"] p {
        color: #8b949e !important;
        font-size: 12px !important;
    }
    div[data-testid="stMetricValue"],
    div[data-testid="metric-container"] div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] > div {
        color: #e6edf3 !important;
        font-size: 22px !important;
        font-weight: 700 !important;
        line-height: 1.2 !important;
    }
    div[data-testid="stMetricDelta"],
    div[data-testid="metric-container"] div[data-testid="stMetricDelta"] {
        font-size: 12px !important;
    }

    h1, h2, h3 { color: #e6edf3 !important; }
    h1 { font-size: 1.6rem !important; font-weight: 700 !important; }

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

    /* Hide Streamlit feedback / rate-us toast and popup */
    div[data-testid="toastContainer"] { display: none !important; }
    div[data-baseweb="toast"]          { display: none !important; }
    .stChatFloatingInputContainer      { display: none !important; }
    button[kind="secondaryFormSubmit"]  { display: none !important; }
    div[data-testid="InputInstructions"]{ display: none !important; }
    section[data-testid="stBottom"]    { display: none !important; }
</style>
""", unsafe_allow_html=True)

_ticker_label = st.session_state.get('ticker_name', '')
_subtitle = f"{_ticker_label} · Stock Analysis &amp; Prediction" if _ticker_label else "Stock Analysis &amp; Prediction"
st.markdown(f"""
<div style="text-align:center; padding: 28px 0 8px 0;">
    <span style="font-size:2.2rem; font-weight:800; letter-spacing:1px;
                 background: linear-gradient(90deg,#e8273b,#ff6b6b);
                 -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
        StockVision
    </span>
    <div style="color:#8b949e; font-size:0.85rem; margin-top:4px; letter-spacing:2px; text-transform:uppercase;">
        {_subtitle}
    </div>
</div>
<hr style="border-color:#21262d; margin:0 0 16px 0;">
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

@st.cache_data(ttl=300, show_spinner=False)
def fetch_live_tsla_data(period: str = "6mo", interval: str = "1d") -> tuple:
    """
    Fetch live TSLA data via yfinance.
    Returns (prepared_df, error_or_None).
    Cached for 5 minutes (ttl=300).
    """
    if not _YF_AVAILABLE:
        return None, "yfinance is not installed. Run: pip install yfinance"
    try:
        raw = yf.download("TSLA", period=period, interval=interval,
                          auto_adjust=True, progress=False)
        if raw is None or len(raw) == 0:
            return None, "yfinance returned no data for TSLA."

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [col[0] if isinstance(col, tuple) else col for col in raw.columns]

        raw = raw.reset_index()

        if 'Datetime' in raw.columns:
            raw = raw.rename(columns={'Datetime': 'Date'})
        elif 'Date' not in raw.columns:
            raw.columns.values[0] = 'Date'

        available = [c for c in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                     if c in raw.columns]
        missing_live = [c for c in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                        if c not in raw.columns]
        if missing_live:
            return None, f"Live data missing columns: {missing_live}"

        raw = raw[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
        raw['Date'] = pd.to_datetime(raw['Date']).dt.tz_localize(None)

        prepared, err = validate_and_prepare(raw)
        return prepared, err

    except Exception as exc:
        return None, f"Failed to fetch live Tesla data: {exc}"


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

def _make_sample_df() -> pd.DataFrame:
    """Shared sample data used by all three sample generators."""
    import datetime
    dates  = pd.date_range(end=datetime.date.today(), periods=120, freq='D')
    np.random.seed(42)
    prices = np.cumprod(1 + np.random.normal(0.001, 0.02, 120)) * 200
    return pd.DataFrame({
        'Date':   dates.strftime('%Y-%m-%d'),
        'Open':   np.round(prices * np.random.uniform(0.98, 1.0, 120), 2),
        'High':   np.round(prices * np.random.uniform(1.0,  1.03, 120), 2),
        'Low':    np.round(prices * np.random.uniform(0.97, 1.0,  120), 2),
        'Close':  np.round(prices, 2),
        'Volume': np.random.randint(20_000_000, 80_000_000, 120),
    })

def generate_sample_csv() -> bytes:
    """Return a sample CSV with the required columns."""
    return _make_sample_df().to_csv(index=False).encode('utf-8')

def generate_sample_json() -> bytes:
    """Return a sample JSON file (records orientation) with the required columns."""
    return _make_sample_df().to_json(orient='records', indent=2).encode('utf-8')

def generate_sample_sql() -> bytes:
    """Return a sample SQL dump (CREATE TABLE + INSERTs) with the required columns."""
    df_s = _make_sample_df()
    lines = [
        "-- Sample stock data — import with the SQL Database source",
        "CREATE TABLE IF NOT EXISTS stock_data (",
        "    Date    TEXT,",
        "    Open    REAL,",
        "    High    REAL,",
        "    Low     REAL,",
        "    Close   REAL,",
        "    Volume  INTEGER",
        ");",
        "",
    ]
    for _, row in df_s.iterrows():
        lines.append(
            f"INSERT INTO stock_data VALUES "
            f"('{row['Date']}', {row['Open']}, {row['High']}, "
            f"{row['Low']}, {row['Close']}, {int(row['Volume'])});"
        )
    return "\n".join(lines).encode('utf-8')



def generate_report_pdf(df, model_bytes, ticker_name="Stock"):
    """
    Generate a polished, professional light-theme PDF report.
    Covers Market Overview (key metrics, price history, drawdown, risk, insights)
    and Prediction (model perf, actual vs predicted, train/test split, next-day forecast).
    Returns raw PDF bytes.
    """
    import io as _io, datetime
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm, mm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, HRFlowable, Image as RLImage, KeepTogether,
    )

    C_RED       = colors.HexColor("#c0392b")      #accent
    C_RED_LIGHT = colors.HexColor("#f8d7da")      #tinted bg
    C_DARK      = colors.HexColor("#1a1a2e")      #headings
    C_BODY      = colors.HexColor("#2c2c2c")      #body text
    C_MUTED     = colors.HexColor("#666666")      #captions / labels
    C_BORDER    = colors.HexColor("#dee2e6")      #table borders
    C_ROW_ALT   = colors.HexColor("#f8f9fa")      #alternating row
    C_ROW_MAIN  = colors.white
    C_HDR_BG    = colors.HexColor("#c0392b")      #table header bg
    C_HDR_TEXT  = colors.white
    C_GREEN     = colors.HexColor("#27ae60")
    C_ORANGE    = colors.HexColor("#e67e22")
    C_BLUE      = colors.HexColor("#2980b9")
    C_CARD_BG   = colors.HexColor("#f0f4ff")      #metric card bg

    W, H = A4
    L_MAR = R_MAR = 1.8*cm
    T_MAR = 3.4*cm   #leave room for header band
    B_MAR = 2.4*cm   #leave room for footer
    inner_w = W - L_MAR - R_MAR
    now_str = datetime.datetime.now().strftime("%d %b %Y  %H:%M")

    def _draw_header_footer(canv, doc):
        canv.saveState()
        band_h = 1.9*cm
        canv.setFillColor(C_RED)
        canv.rect(0, H - band_h, W, band_h, fill=1, stroke=0)
        canv.setFillColor(colors.white)
        canv.setFont("Helvetica-Bold", 14)
        canv.drawString(L_MAR, H - band_h + 0.65*cm, "StockVision")
        canv.setFont("Helvetica", 8)
        canv.drawRightString(W - R_MAR, H - band_h + 1.05*cm,
                             f"{ticker_name}  |  Stock Analysis Report")
        canv.drawRightString(W - R_MAR, H - band_h + 0.35*cm, now_str)
        canv.setFillColor(C_MUTED)
        canv.setFont("Helvetica", 7)
        footer_y = 1.1*cm
        canv.drawString(L_MAR, footer_y,
                        "Generated by StockVision  |  For informational purposes only — not investment advice.")
        canv.drawRightString(W - R_MAR, footer_y, f"Page {doc.page}")
        canv.setStrokeColor(C_BORDER)
        canv.setLineWidth(0.5)
        canv.line(L_MAR, footer_y + 0.45*cm, W - R_MAR, footer_y + 0.45*cm)
        canv.restoreState()

    buf = _io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=L_MAR, rightMargin=R_MAR,
        topMargin=T_MAR, bottomMargin=B_MAR,
    )

    ss = getSampleStyleSheet()

    def S(name, parent="Normal", **kw):
        return ParagraphStyle(name, parent=ss[parent], **kw)

    ST_COVER_TITLE = S("CTitle", fontSize=36, textColor=C_RED,
                       fontName="Helvetica-Bold", alignment=TA_CENTER,
                       leading=44, spaceAfter=14, spaceBefore=0)
    ST_COVER_SUB   = S("CSub",   fontSize=13, textColor=C_MUTED,
                       alignment=TA_CENTER, leading=18, spaceBefore=6, spaceAfter=8)
    ST_COVER_BODY  = S("CBody",  fontSize=10, textColor=C_BODY,
                       alignment=TA_CENTER, leading=16, spaceAfter=4)

    ST_H1    = S("H1",    fontSize=17, textColor=C_DARK, fontName="Helvetica-Bold",
                 spaceBefore=6, spaceAfter=4, leading=22)
    ST_H2    = S("H2",    fontSize=12, textColor=C_RED,  fontName="Helvetica-Bold",
                 spaceBefore=10, spaceAfter=3)
    ST_BODY  = S("Body",  fontSize=9,  textColor=C_BODY, leading=15, spaceAfter=4)
    ST_CAP   = S("Cap",   fontSize=8,  textColor=C_MUTED, alignment=TA_CENTER,
                 spaceBefore=4, spaceAfter=10, fontName="Helvetica-Oblique")
    ST_CELL  = S("Cell",  fontSize=9,  textColor=C_BODY)
    ST_CELLB = S("CellB", fontSize=9,  textColor=C_BODY, fontName="Helvetica-Bold")
    ST_LBL   = S("Lbl",   fontSize=8,  textColor=C_MUTED, alignment=TA_CENTER, spaceAfter=2)
    ST_VAL   = S("Val",   fontSize=16, textColor=C_DARK, fontName="Helvetica-Bold",
                 alignment=TA_CENTER, leading=20)
    ST_DELTA_G = S("DG",  fontSize=8, textColor=C_GREEN, alignment=TA_CENTER, spaceAfter=2)
    ST_DELTA_R = S("DR",  fontSize=8, textColor=colors.HexColor("#c0392b"), alignment=TA_CENTER, spaceAfter=2)
    ST_FOOT  = S("Foot",  fontSize=7, textColor=C_MUTED, alignment=TA_CENTER)

    def hr(color=C_BORDER, thickness=0.8, space_before=0, space_after=8):
        return HRFlowable(width="100%", thickness=thickness, color=color,
                          spaceBefore=space_before, spaceAfter=space_after)

    def section_bar(title):
        """Red-background section title bar."""
        tbl = Table([[Paragraph(f'<font color="white"><b>{title}</b></font>',
                                S("SB", fontSize=11, textColor=colors.white,
                                  fontName="Helvetica-Bold", leading=14))]],
                    colWidths=[inner_w])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), C_RED),
            ("TOPPADDING",    (0,0), (-1,-1), 8),
            ("BOTTOMPADDING", (0,0), (-1,-1), 8),
            ("LEFTPADDING",   (0,0), (-1,-1), 14),
            ("RIGHTPADDING",  (0,0), (-1,-1), 14),
        ]))
        return tbl

    def fig_to_rl(fig, w_cm=15, h_cm=7):
        ib = _io.BytesIO()
        fig.savefig(ib, format="png", dpi=160, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close(fig)
        ib.seek(0)
        return RLImage(ib, width=w_cm*cm, height=h_cm*cm)

    def metric_cards(items, cols=4):
        """items = list of (label, value, delta_or_None)"""
        cw = inner_w / cols
        cells = []
        for label, value, delta in items:
            if delta:
                is_pos = not delta.startswith("-")
                dp = Paragraph(delta, ST_DELTA_G if is_pos else ST_DELTA_R)
            else:
                dp = Paragraph("", ST_LBL)
            inner = Table(
                [[Paragraph(value, ST_VAL)], [dp], [Paragraph(label, ST_LBL)]],
                colWidths=[cw - 12]
            )
            inner.setStyle(TableStyle([
                ("ALIGN",         (0,0), (-1,-1), "CENTER"),
                ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
                ("TOPPADDING",    (0,0), (-1,-1), 2),
                ("BOTTOMPADDING", (0,0), (-1,-1), 2),
                ("LEFTPADDING",   (0,0), (-1,-1), 0),
                ("RIGHTPADDING",  (0,0), (-1,-1), 0),
            ]))
            cells.append(inner)
        t = Table([cells], colWidths=[cw]*cols)
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), C_CARD_BG),
            ("GRID",          (0,0), (-1,-1), 0.6, C_BORDER),
            ("TOPPADDING",    (0,0), (-1,-1), 12),
            ("BOTTOMPADDING", (0,0), (-1,-1), 12),
            ("LEFTPADDING",   (0,0), (-1,-1), 6),
            ("RIGHTPADDING",  (0,0), (-1,-1), 6),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
            ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ]))
        return t

    def data_table(headers, rows, col_widths=None):
        """Styled data table with red header row."""
        if col_widths is None:
            col_widths = [inner_w/len(headers)] * len(headers)
        header_row = [Paragraph(f"<b>{h}</b>",
                                S("TH", fontSize=9, textColor=colors.white,
                                  fontName="Helvetica-Bold")) for h in headers]
        data_rows = [[Paragraph(str(c), ST_CELL) for c in row] for row in rows]
        t = Table([header_row] + data_rows, colWidths=col_widths, repeatRows=1)
        style = [
            ("BACKGROUND",    (0,0),  (-1,0),  C_RED),
            ("TEXTCOLOR",     (0,0),  (-1,0),  colors.white),
            ("FONTNAME",      (0,0),  (-1,0),  "Helvetica-Bold"),
            ("GRID",          (0,0),  (-1,-1), 0.5, C_BORDER),
            ("TOPPADDING",    (0,0),  (-1,-1), 6),
            ("BOTTOMPADDING", (0,0),  (-1,-1), 6),
            ("LEFTPADDING",   (0,0),  (-1,-1), 8),
            ("FONTSIZE",      (0,0),  (-1,-1), 9),
            ("VALIGN",        (0,0),  (-1,-1), "MIDDLE"),
        ]
        for i, _ in enumerate(data_rows):
            bg = C_ROW_ALT if i % 2 == 0 else C_ROW_MAIN
            style.append(("BACKGROUND", (0, i+1), (-1, i+1), bg))
        t.setStyle(TableStyle(style))
        return t

    def mpl_style(ax, fig):
        """Apply clean white/light style to matplotlib axes."""
        fig.patch.set_facecolor("white")
        ax.set_facecolor("#f9f9fb")
        ax.tick_params(colors="#555555", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#cccccc")
            sp.set_linewidth(0.6)
        ax.grid(color="#e0e0e0", lw=0.5, linestyle="--")
        ax.yaxis.label.set_color("#555555")
        ax.xaxis.label.set_color("#555555")

    rm = risk_metrics(df)
    pred_available = False
    if model_bytes is not None:
        try:
            model_data   = engineer_features(df)
            model, X_train, X_test, y_train, y_test, preds, m_metrics, feature_cols = \
                evaluate_model(model_data, model_bytes)
            last_close   = float(df["Close"].iloc[-1])
            next_pred    = model.predict(
                model_data[feature_cols].iloc[-1].values.reshape(1, -1))[0]
            change       = next_pred - last_close
            pct_change   = change / last_close * 100
            pred_available = True
        except Exception:
            pred_available = False

    story = []

    story.append(Spacer(1, 1.5*cm))
    story.append(Paragraph("StockVision", ST_COVER_TITLE))
    story.append(Paragraph(f"{ticker_name} &mdash; Stock Analysis &amp; Prediction Report",
                            ST_COVER_SUB))
    story.append(Spacer(1, 0.3*cm))
    story.append(hr(C_RED, thickness=2, space_after=20))

    cover_rows = [
        ["Report Date",       now_str],
        ["Dataset Range",     f"{df.index.min().date()}  \u2192  {df.index.max().date()}"],
        ["Total Trading Days",f"{len(df):,}"],
        ["Model",             "Linear Regression (pre-trained .pkl)"],
        ["Sections",          "Market Overview  |  Risk Analysis  |  Model Prediction"],
    ]
    ctbl = Table(cover_rows, colWidths=[4.5*cm, inner_w - 4.5*cm])
    ctbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (0,-1), C_ROW_ALT),
        ("BACKGROUND",    (1,0), (1,-1), C_ROW_MAIN),
        ("TEXTCOLOR",     (0,0), (0,-1), C_MUTED),
        ("TEXTCOLOR",     (1,0), (1,-1), C_BODY),
        ("FONTNAME",      (0,0), (0,-1), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 9),
        ("GRID",          (0,0), (-1,-1), 0.5, C_BORDER),
        ("TOPPADDING",    (0,0), (-1,-1), 9),
        ("BOTTOMPADDING", (0,0), (-1,-1), 9),
        ("LEFTPADDING",   (0,0), (-1,-1), 12),
    ]))
    story.append(ctbl)
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph(
        "This report was automatically generated by <b>StockVision</b>. "
        f"It provides a comprehensive analysis of <b>{ticker_name}</b> stock data, "
        "covering key performance metrics, price history, volatility and risk indicators, "
        + ("and a Linear Regression model evaluation with next-day price forecasting."
           if pred_available else
           "and risk analysis. The prediction section is unavailable (model file not loaded)."),
        ST_COVER_BODY))
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph(
        "<b>Disclaimer:</b> This report is for informational purposes only and does not "
        "constitute investment advice. Past performance is not indicative of future results.",
        S("Disc", fontSize=8, textColor=C_MUTED, alignment=TA_CENTER,
          fontName="Helvetica-Oblique")))
    story.append(PageBreak())
    story.append(Paragraph("1. Market Overview", ST_H1))
    story.append(hr(C_RED, thickness=1.5, space_after=10))

    story.append(section_bar("1.1  Key Performance Metrics"))
    story.append(Spacer(1, 0.3*cm))

    latest_close = float(df["Close"].iloc[-1])
    prev_close   = float(df["Close"].iloc[-2])
    day_chg      = (latest_close / prev_close - 1) * 100

    row1 = [
        ("Latest Close",       f"${latest_close:,.2f}", f"{day_chg:+.2f}%"),
        ("Total Return",       f"{rm['Total Return (%)']:,.1f}%", None),
        ("Sharpe Ratio",       f"{rm['Sharpe Ratio']:.3f}", None),
        ("Max Drawdown",       f"{rm['Max Drawdown (%)']:.1f}%", None),
    ]
    row2 = [
        ("Ann. Return",        f"{rm['Annualized Return (%)']:.2f}%", None),
        ("Ann. Volatility",    f"{rm['Annualized Volatility (%)']:.2f}%", None),
        ("Best Single Day",    f"{rm['Best Day (%)']:.2f}%", None),
        ("Positive Days",      f"{rm['Positive Days (%)']:.1f}%", None),
    ]
    story.append(metric_cards(row1, cols=4))
    story.append(Spacer(1, 0.25*cm))
    story.append(metric_cards(row2, cols=4))
    story.append(Spacer(1, 0.5*cm))

    story.append(section_bar("1.2  Price History with Moving Averages"))
    story.append(Spacer(1, 0.25*cm))

    df_p = df.copy()
    df_p["MA20"]  = df_p["Close"].rolling(20).mean()
    df_p["MA50"]  = df_p["Close"].rolling(50).mean()
    df_p["MA200"] = df_p["Close"].rolling(200).mean()

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(df_p.index, df_p["Close"],  color="#c0392b", lw=1.4, label="Close Price", zorder=3)
    ax.plot(df_p.index, df_p["MA20"],   color="#e67e22", lw=0.9, ls="--", label="MA 20", zorder=2)
    ax.plot(df_p.index, df_p["MA50"],   color="#2980b9", lw=0.9, ls="--", label="MA 50", zorder=2)
    ax.plot(df_p.index, df_p["MA200"],  color="#27ae60", lw=1.1,           label="MA 200",zorder=2)
    mpl_style(ax, fig)
    ax.set_ylabel("Price (USD)", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(fontsize=8, framealpha=0.9, loc="upper left",
              edgecolor="#cccccc", fancybox=False)
    fig.tight_layout(pad=0.5)
    story.append(fig_to_rl(fig, w_cm=16, h_cm=6.5))
    story.append(Paragraph(
        f"Figure 1 — {ticker_name} closing price with 20-, 50-, and 200-day simple moving averages (full dataset)",
        ST_CAP))
    story.append(PageBreak())

    story.append(section_bar("1.3  Drawdown from All-Time High"))
    story.append(Spacer(1, 0.25*cm))

    roll_max = df["Close"].cummax()
    drawdown = (df["Close"] - roll_max) / roll_max * 100

    fig2, ax2 = plt.subplots(figsize=(13, 3.5))
    ax2.fill_between(drawdown.index, drawdown, 0, color="#c0392b", alpha=0.25, label="Drawdown")
    ax2.plot(drawdown.index, drawdown, color="#c0392b", lw=0.9)
    mpl_style(ax2, fig2)
    ax2.set_ylabel("Drawdown (%)", fontsize=9)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    ax2.axhline(0, color="#888888", lw=0.6)
    fig2.tight_layout(pad=0.5)
    story.append(fig_to_rl(fig2, w_cm=16, h_cm=5))
    story.append(Paragraph("Figure 2 — Rolling drawdown (%) from the historical all-time high", ST_CAP))
    story.append(Spacer(1, 0.4*cm))

    story.append(section_bar("1.4  Risk & Return Summary"))
    story.append(Spacer(1, 0.25*cm))

    risk_headers = ["Metric", "Value"]
    risk_rows_clean = [[k, str(v)] for k, v in rm.items()]
    story.append(data_table(risk_headers, risk_rows_clean,
                            col_widths=[inner_w*0.55, inner_w*0.45]))
    story.append(Spacer(1, 0.5*cm))

    story.append(section_bar("1.5  Key Insights"))
    story.append(Spacer(1, 0.25*cm))

    insights = [
        ("Total Return",
         f"The dataset delivered a total return of <b>{rm['Total Return (%)']:,.1f}%</b> "
         f"across the full dataset period."),
        ("Volatility",
         f"Annualized volatility of <b>{rm['Annualized Volatility (%)']:.1f}%</b> "
         f"classifies this as a {'high' if rm['Annualized Volatility (%)'] > 30 else 'moderate'}-volatility asset."),
        ("Sharpe Ratio",
         f"A Sharpe ratio of <b>{rm['Sharpe Ratio']:.3f}</b> indicates that returns "
         f"reasonably compensate for the risk taken."),
        ("Max Drawdown",
         f"Maximum drawdown of <b>{rm['Max Drawdown (%)']:.1f}%</b> highlights the "
         f"downside exposure during correction periods."),
        ("Positive Days",
         f"The asset closed higher on <b>{rm['Positive Days (%)']:.1f}%</b> of all trading days "
         f"in the dataset."),
    ]
    ins_rows = [[Paragraph(f"<b>{t}</b>", ST_CELLB), Paragraph(b, ST_BODY)]
                for t, b in insights]
    itbl = Table(ins_rows, colWidths=[3.2*cm, inner_w - 3.2*cm])
    style_ins = [
        ("GRID",          (0,0), (-1,-1), 0.5, C_BORDER),
        ("TOPPADDING",    (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("LEFTPADDING",   (0,0), (-1,-1), 10),
        ("VALIGN",        (0,0), (-1,-1), "TOP"),
        ("TEXTCOLOR",     (0,0), (0,-1),  C_RED),
    ]
    for i in range(len(ins_rows)):
        style_ins.append(("BACKGROUND", (0,i), (-1,i),
                          C_ROW_ALT if i % 2 == 0 else C_ROW_MAIN))
    itbl.setStyle(TableStyle(style_ins))
    story.append(itbl)

    if not pred_available:
        story.append(Spacer(1, 1*cm))
        story.append(Paragraph(
            "&#9888;  Prediction section is unavailable — "
            "no model .pkl was loaded.",
            S("NoModel", fontSize=9, textColor=C_ORANGE)))
        doc.build(story, onFirstPage=_draw_header_footer, onLaterPages=_draw_header_footer)
        buf.seek(0)
        return buf.read()

    story.append(PageBreak())

    story.append(Paragraph("2. Prediction — Linear Regression Model", ST_H1))
    story.append(hr(C_RED, thickness=1.5, space_after=10))

    story.append(section_bar("2.1  Model Information"))
    story.append(Spacer(1, 0.3*cm))

    info_rows = [["Algorithm", "Linear Regression"],
                 ["Features",  str(model.n_features_in_)],
                 ["Training Rows", f"{len(X_train):,}  (80%)"],
                 ["Test Rows",     f"{len(X_test):,}  (20%)"],
                 ["Split Strategy","Time-ordered (no shuffle)"]]
    info_tbl = Table(info_rows, colWidths=[inner_w*0.4, inner_w*0.6])
    style_info = [
        ("GRID",          (0,0), (-1,-1), 0.5, C_BORDER),
        ("FONTNAME",      (0,0), (0,-1),  "Helvetica-Bold"),
        ("TEXTCOLOR",     (0,0), (0,-1),  C_MUTED),
        ("TOPPADDING",    (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("LEFTPADDING",   (0,0), (-1,-1), 10),
        ("FONTSIZE",      (0,0), (-1,-1), 9),
    ]
    for i in range(len(info_rows)):
        style_info.append(("BACKGROUND", (0,i), (-1,i),
                           C_ROW_ALT if i%2==0 else C_ROW_MAIN))
    info_tbl.setStyle(TableStyle(style_info))
    story.append(info_tbl)
    story.append(Spacer(1, 0.5*cm))

    story.append(section_bar("2.2  Model Performance Metrics"))
    story.append(Spacer(1, 0.3*cm))
    perf_items = [
        ("Mean Abs. Error (MAE)",    f"${m_metrics['MAE']:.2f}", None),
        ("Root Mean Sq. Error (RMSE)",f"${m_metrics['RMSE']:.2f}",None),
        ("R\u00b2 Score",            f"{m_metrics['R2']:.4f}",   None),
    ]
    story.append(metric_cards(perf_items, cols=3))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(
        f"The model explains <b>{m_metrics['R2']*100:.1f}%</b> of the variance in "
        f"next-day closing price. The average daily prediction error is "
        f"<b>${m_metrics['MAE']:.2f}</b> (MAE) with an RMSE of "
        f"<b>${m_metrics['RMSE']:.2f}</b>.",
        ST_BODY))
    story.append(Spacer(1, 0.5*cm))

    story.append(section_bar("2.3  Actual vs Predicted — Test Set"))
    story.append(Spacer(1, 0.25*cm))

    fig3, ax3 = plt.subplots(figsize=(13, 4.5))
    ax3.plot(y_test.index, y_test.values, color="#2980b9", lw=1.4, label="Actual Close")
    ax3.plot(y_test.index, preds,         color="#c0392b", lw=1.0, ls="--",
             label="Predicted Close", alpha=0.85)
    mpl_style(ax3, fig3)
    ax3.set_ylabel("Price (USD)", fontsize=9)
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax3.legend(fontsize=8, framealpha=0.9, edgecolor="#cccccc", fancybox=False)
    fig3.tight_layout(pad=0.5)
    story.append(fig_to_rl(fig3, w_cm=16, h_cm=6))
    story.append(Paragraph(
        "Figure 3 — Actual vs predicted next-day close price on the held-out test set (most recent 20% of data)",
        ST_CAP))
    story.append(PageBreak())

    story.append(section_bar("2.4  Train / Test Split"))
    story.append(Spacer(1, 0.25*cm))

    fig4, ax4 = plt.subplots(figsize=(13, 4))
    ax4.plot(y_train.index, y_train.values, color="#2980b9", lw=1.0,
             label=f"Training set  ({len(y_train):,} rows, 80%)")
    ax4.plot(y_test.index,  y_test.values,  color="#27ae60", lw=1.0,
             label=f"Test set  ({len(y_test):,} rows, 20%)")
    ax4.axvline(y_test.index[0], color="#c0392b", ls="--", lw=1.0, label="Split point")
    ax4.fill_betweenx([y_train.values.min(), y_train.values.max()],
                      y_train.index[0], y_test.index[0],
                      alpha=0.05, color="#2980b9")
    ax4.fill_betweenx([y_test.values.min(), y_test.values.max()],
                      y_test.index[0], y_test.index[-1],
                      alpha=0.05, color="#27ae60")
    mpl_style(ax4, fig4)
    ax4.set_ylabel("Close Price (USD)", fontsize=9)
    ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax4.legend(fontsize=8, framealpha=0.9, edgecolor="#cccccc", fancybox=False)
    fig4.tight_layout(pad=0.5)
    story.append(fig_to_rl(fig4, w_cm=16, h_cm=5.5))
    story.append(Paragraph(
        "Figure 4 — Time-ordered 80/20 train-test split; no data leakage (shuffle=False)",
        ST_CAP))
    story.append(Spacer(1, 0.5*cm))

    story.append(section_bar("2.5  Next-Day Price Forecast"))
    story.append(Spacer(1, 0.3*cm))

    delta_sign = "+" if change >= 0 else ""
    forecast_items = [
        ("Current Close Price",       f"${last_close:,.2f}", None),
        ("Predicted Next-Day Close",  f"${next_pred:,.2f}",
         f"{delta_sign}{change:.2f}  ({delta_sign}{pct_change:.2f}%)"),
        ("Model R\u00b2 Score",       f"{m_metrics['R2']:.4f}", None),
    ]
    story.append(metric_cards(forecast_items, cols=3))
    story.append(Spacer(1, 0.4*cm))

    direction = "an increase" if change >= 0 else "a decrease"
    story.append(Paragraph(
        f"The pre-trained Linear Regression model — using lag features (Close Lag 1–5) "
        f"and rolling statistics (7- and 30-day mean/std) — forecasts a next-day closing "
        f"price of <b>${next_pred:,.2f}</b>, representing {direction} of "
        f"<b>${abs(change):.2f} ({abs(pct_change):.2f}%)</b> from the most recent "
        f"closing price of <b>${last_close:,.2f}</b>.",
        ST_BODY))

    story.append(Spacer(1, 1.5*cm))
    story.append(hr(C_BORDER, thickness=0.8))
    story.append(Spacer(1, 0.2*cm))
    story.append(Paragraph(
        f"End of Report  \u2014  StockVision  \u00b7  {ticker_name} Analysis  \u00b7  "
        f"Generated {now_str}",
        ST_FOOT))

    doc.build(story, onFirstPage=_draw_header_footer, onLaterPages=_draw_header_footer)
    buf.seek(0)
    return buf.read()

def generate_live_tesla_pdf(live_df, model_bytes, live_period="6mo", live_interval="1d"):
    """
    Generate a polished PDF report for the Live Tesla (TSLA) tab.
    Sections:
      1. Live Market Snapshot   — KPI cards
      2. Tesla Live Price Trend — close price + MA20 chart
      3. Next-Day Prediction    — metrics, Buy/Sell signal
      4. Recent Trading Data    — last 10 trading days table
      5. Volume Activity        — 30-day volume bar chart
    Returns raw PDF bytes.
    """
    import io as _io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import matplotlib.patches as mpatches
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import cm
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, HRFlowable, Image as RLImage, KeepTogether,
    )

    C_RED       = colors.HexColor("#c0392b")
    C_RED_LIGHT = colors.HexColor("#f8d7da")
    C_DARK      = colors.HexColor("#1a1a2e")
    C_BODY      = colors.HexColor("#2c2c2c")
    C_MUTED     = colors.HexColor("#666666")
    C_BORDER    = colors.HexColor("#dee2e6")
    C_ROW_ALT   = colors.HexColor("#f8f9fa")
    C_ROW_MAIN  = colors.white
    C_GREEN     = colors.HexColor("#27ae60")
    C_ORANGE    = colors.HexColor("#e67e22")
    C_CARD_BG   = colors.HexColor("#f0f4ff")

    W, H = A4
    L_MAR = R_MAR = 1.8 * cm
    T_MAR = 3.4 * cm
    B_MAR = 2.4 * cm
    inner_w = W - L_MAR - R_MAR
    now_str = datetime.datetime.now().strftime("%d %b %Y  %H:%M")
    ticker_name = "TESLA (TSLA) — LIVE"

    def _draw_hf(canv, doc):
        canv.saveState()
        band_h = 1.9 * cm
        canv.setFillColor(C_RED)
        canv.rect(0, H - band_h, W, band_h, fill=1, stroke=0)
        canv.setFillColor(colors.white)
        canv.setFont("Helvetica-Bold", 14)
        canv.drawString(L_MAR, H - band_h + 0.65 * cm, "StockVision")
        canv.setFont("Helvetica", 8)
        canv.drawRightString(W - R_MAR, H - band_h + 1.05 * cm,
                             f"{ticker_name}  |  Live Analysis Report")
        canv.drawRightString(W - R_MAR, H - band_h + 0.35 * cm, now_str)
        canv.setFillColor(C_MUTED)
        canv.setFont("Helvetica", 7)
        footer_y = 1.1 * cm
        canv.drawString(L_MAR, footer_y,
                        "Generated by StockVision  |  For informational purposes only — not investment advice.")
        canv.drawRightString(W - R_MAR, footer_y, f"Page {doc.page}")
        canv.setStrokeColor(C_BORDER)
        canv.setLineWidth(0.5)
        canv.line(L_MAR, footer_y + 0.45 * cm, W - R_MAR, footer_y + 0.45 * cm)
        canv.restoreState()

    buf = _io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=L_MAR, rightMargin=R_MAR,
                            topMargin=T_MAR, bottomMargin=B_MAR)

    ss = getSampleStyleSheet()
    def S(name, parent="Normal", **kw):
        return ParagraphStyle(name, parent=ss[parent], **kw)

    ST_COVER_TITLE = S("CT",  fontSize=34, textColor=C_RED,   fontName="Helvetica-Bold",
                        alignment=TA_CENTER, leading=42, spaceAfter=10)
    ST_COVER_SUB   = S("CS",  fontSize=13, textColor=C_MUTED, alignment=TA_CENTER,
                        leading=18, spaceAfter=6)
    ST_COVER_BODY  = S("CB",  fontSize=10, textColor=C_BODY,  alignment=TA_CENTER,
                        leading=15, spaceAfter=4)
    ST_H1    = S("H1",  fontSize=17, textColor=C_DARK, fontName="Helvetica-Bold",
                  spaceBefore=6, spaceAfter=4, leading=22)
    ST_BODY  = S("BO",  fontSize=9,  textColor=C_BODY, leading=15, spaceAfter=4)
    ST_CAP   = S("CA",  fontSize=8,  textColor=C_MUTED, alignment=TA_CENTER,
                  spaceBefore=4, spaceAfter=10, fontName="Helvetica-Oblique")
    ST_CELL  = S("CE",  fontSize=9,  textColor=C_BODY)
    ST_CELLB = S("CEB", fontSize=9,  textColor=C_BODY, fontName="Helvetica-Bold")
    ST_LBL   = S("LB",  fontSize=8,  textColor=C_MUTED, alignment=TA_CENTER, spaceAfter=2)
    ST_VAL   = S("VA",  fontSize=15, textColor=C_DARK, fontName="Helvetica-Bold",
                  alignment=TA_CENTER, leading=19)
    ST_DELTA_G = S("DG", fontSize=8, textColor=C_GREEN,  alignment=TA_CENTER, spaceAfter=2)
    ST_DELTA_R = S("DR", fontSize=8, textColor=C_RED,    alignment=TA_CENTER, spaceAfter=2)
    ST_FOOT  = S("FO",  fontSize=7,  textColor=C_MUTED,  alignment=TA_CENTER)
    ST_SIG_BUY  = S("SB", fontSize=20, textColor=C_GREEN, fontName="Helvetica-Bold",
                     alignment=TA_CENTER, leading=26)
    ST_SIG_SELL = S("SS", fontSize=20, textColor=C_RED,   fontName="Helvetica-Bold",
                     alignment=TA_CENTER, leading=26)

    def hr(color=C_BORDER, thickness=0.8, space_before=0, space_after=8):
        return HRFlowable(width="100%", thickness=thickness, color=color,
                          spaceBefore=space_before, spaceAfter=space_after)

    def section_bar(title):
        tbl = Table([[Paragraph(f'<font color="white"><b>{title}</b></font>',
                                S("SBr", fontSize=11, textColor=colors.white,
                                  fontName="Helvetica-Bold", leading=14))]],
                    colWidths=[inner_w])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), C_RED),
            ("TOPPADDING",    (0,0), (-1,-1), 8),
            ("BOTTOMPADDING", (0,0), (-1,-1), 8),
            ("LEFTPADDING",   (0,0), (-1,-1), 14),
        ]))
        return tbl

    def metric_cards(items, cols=4):
        cw = inner_w / cols
        cells = []
        for label, value, delta in items:
            if delta:
                is_pos = not delta.startswith("-")
                dp = Paragraph(delta, ST_DELTA_G if is_pos else ST_DELTA_R)
            else:
                dp = Paragraph("", ST_LBL)
            inner = Table(
                [[Paragraph(value, ST_VAL)], [dp], [Paragraph(label, ST_LBL)]],
                colWidths=[cw - 12])
            inner.setStyle(TableStyle([
                ("ALIGN",  (0,0), (-1,-1), "CENTER"),
                ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
                ("TOPPADDING",    (0,0), (-1,-1), 2),
                ("BOTTOMPADDING", (0,0), (-1,-1), 2),
                ("LEFTPADDING",   (0,0), (-1,-1), 0),
                ("RIGHTPADDING",  (0,0), (-1,-1), 0),
            ]))
            cells.append(inner)
        t = Table([cells], colWidths=[cw] * cols)
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), C_CARD_BG),
            ("GRID",          (0,0), (-1,-1), 0.6, C_BORDER),
            ("TOPPADDING",    (0,0), (-1,-1), 12),
            ("BOTTOMPADDING", (0,0), (-1,-1), 12),
            ("LEFTPADDING",   (0,0), (-1,-1), 6),
            ("RIGHTPADDING",  (0,0), (-1,-1), 6),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
            ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ]))
        return t

    def data_table(headers, rows, col_widths=None):
        if col_widths is None:
            col_widths = [inner_w / len(headers)] * len(headers)
        header_row = [Paragraph(f"<b>{h}</b>",
                                S("TH2", fontSize=9, textColor=colors.white,
                                  fontName="Helvetica-Bold")) for h in headers]
        data_rows  = [[Paragraph(str(c), ST_CELL) for c in row] for row in rows]
        t = Table([header_row] + data_rows, colWidths=col_widths, repeatRows=1)
        style = [
            ("BACKGROUND",    (0,0), (-1,0),  C_RED),
            ("TEXTCOLOR",     (0,0), (-1,0),  colors.white),
            ("FONTNAME",      (0,0), (-1,0),  "Helvetica-Bold"),
            ("GRID",          (0,0), (-1,-1), 0.5, C_BORDER),
            ("TOPPADDING",    (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
            ("LEFTPADDING",   (0,0), (-1,-1), 8),
            ("FONTSIZE",      (0,0), (-1,-1), 9),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ]
        for i in range(len(data_rows)):
            bg = C_ROW_ALT if i % 2 == 0 else C_ROW_MAIN
            style.append(("BACKGROUND", (0, i+1), (-1, i+1), bg))
        t.setStyle(TableStyle(style))
        return t

    def fig_to_rl(fig, w_cm=15, h_cm=7):
        ib = _io.BytesIO()
        fig.savefig(ib, format="png", dpi=160, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close(fig)
        ib.seek(0)
        return RLImage(ib, width=w_cm * cm, height=h_cm * cm)

    def mpl_style(ax, fig):
        fig.patch.set_facecolor("white")
        ax.set_facecolor("#f9f9fb")
        ax.tick_params(colors="#555555", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#cccccc")
            sp.set_linewidth(0.6)
        ax.grid(color="#e0e0e0", lw=0.5, linestyle="--")
        ax.yaxis.label.set_color("#555555")
        ax.xaxis.label.set_color("#555555")

    last_close  = float(live_df['Close'].iloc[-1])
    prev_close  = float(live_df['Close'].iloc[-2]) if len(live_df) > 1 else last_close
    day_chg     = last_close - prev_close
    day_pct     = (day_chg / prev_close * 100) if prev_close else 0
    period_high = float(live_df['Close'].max())
    period_low  = float(live_df['Close'].min())
    avg_vol     = live_df['Volume'].mean()
    data_from   = live_df.index.min().date()
    data_to     = live_df.index.max().date()

    pred_available = False
    pred_next = change = pct_change = 0.0
    signal = "N/A"
    m_r2   = None
    if model_bytes is not None and len(live_df) >= 60:
        try:
            _lm   = pickle.loads(model_bytes)
            _feats = list(_lm.feature_names_in_)
            _md   = engineer_features(live_df)
            if all(f in _md.columns for f in _feats):
                _row      = _md[_feats].iloc[-1].values.reshape(1, -1)
                pred_next = float(_lm.predict(_row)[0])
                change    = pred_next - last_close
                pct_change = change / last_close * 100
                signal    = "BUY" if pred_next > last_close else "SELL"
                m_r2      = None  
                pred_available = True
        except Exception:
            pred_available = False

    story = []

    story.append(Spacer(1, 1.8 * cm))
    story.append(Paragraph("StockVision", ST_COVER_TITLE))
    story.append(Paragraph("Live Tesla (TSLA) Analysis Report", ST_COVER_SUB))
    story.append(hr(C_RED, thickness=1.5, space_before=6, space_after=12))
    story.append(Paragraph(f"Data period: <b>{live_period}</b> &nbsp;|&nbsp; "
                            f"Interval: <b>{live_interval}</b> &nbsp;|&nbsp; "
                            f"Rows: <b>{len(live_df):,}</b>", ST_COVER_BODY))
    story.append(Paragraph(f"Coverage: <b>{data_from}</b> → <b>{data_to}</b>", ST_COVER_BODY))
    story.append(Paragraph(f"Generated: <b>{now_str}</b>", ST_COVER_BODY))
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph(
        "This report provides a real-time snapshot of Tesla Inc. (TSLA) stock data "
        "fetched via Yahoo Finance, combined with next-day price prediction from a "
        "pre-trained Linear Regression model. For informational purposes only — "
        "not investment advice.",
        ST_BODY))
    story.append(PageBreak())

    story.append(Paragraph("1. Live Market Snapshot", ST_H1))
    story.append(hr(C_RED, thickness=1.5, space_after=10))
    story.append(section_bar("1.1  Key Price Metrics"))
    story.append(Spacer(1, 0.3 * cm))

    day_delta_str = f"{day_chg:+.2f} ({day_pct:+.2f}%)"
    kpi_items = [
        ("Latest Close",     f"${last_close:,.2f}",   day_delta_str),
        ("Period High",      f"${period_high:,.2f}",  None),
        ("Period Low",       f"${period_low:,.2f}",   None),
        ("Avg Daily Volume", f"{avg_vol/1e6:.1f}M",   None),
    ]
    story.append(metric_cards(kpi_items, cols=4))
    story.append(Spacer(1, 0.5 * cm))

    story.append(section_bar("1.2  Risk & Return Summary"))
    story.append(Spacer(1, 0.3 * cm))
    _rm = risk_metrics(live_df)
    risk_rows = [[k, str(v)] for k, v in _rm.items()]
    story.append(data_table(["Metric", "Value"], risk_rows,
                             col_widths=[inner_w * 0.6, inner_w * 0.4]))
    story.append(Spacer(1, 0.5 * cm))

    story.append(PageBreak())
    story.append(Paragraph("2. Tesla Live Price Trend", ST_H1))
    story.append(hr(C_RED, thickness=1.5, space_after=10))
    story.append(section_bar("2.1  Close Price with 20-Day Moving Average"))
    story.append(Spacer(1, 0.25 * cm))

    fig1, ax1 = plt.subplots(figsize=(13, 4.5))
    ax1.fill_between(live_df.index, live_df['Close'], alpha=0.12, color="#c0392b")
    ax1.plot(live_df.index, live_df['Close'], color="#c0392b", lw=1.8, label="Close Price")
    _ma20 = live_df['Close'].rolling(20).mean()
    ax1.plot(live_df.index, _ma20, color="#e67e22", lw=1.2, ls="--", label="MA 20")
    mpl_style(ax1, fig1)
    ax1.set_ylabel("Price (USD)", fontsize=9)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.legend(fontsize=8, framealpha=0.9, edgecolor="#cccccc", fancybox=False)
    fig1.tight_layout(pad=0.5)
    story.append(fig_to_rl(fig1, w_cm=16, h_cm=5.5))
    story.append(Paragraph(
        f"Figure 1 — TSLA daily close price with 20-day moving average "
        f"({data_from} to {data_to})", ST_CAP))
    story.append(Spacer(1, 0.4 * cm))

    story.append(section_bar("2.2  Daily Returns Distribution"))
    story.append(Spacer(1, 0.25 * cm))

    _rets = live_df['Close'].pct_change().dropna() * 100
    fig2, ax2 = plt.subplots(figsize=(13, 3.5))
    ax2.hist(_rets, bins=30, color="#c0392b", alpha=0.7, edgecolor="white", linewidth=0.5)
    ax2.axvline(0,             color="#555555", lw=1.0, ls="--")
    ax2.axvline(_rets.mean(),  color="#e67e22", lw=1.2, ls="-",  label=f"Mean {_rets.mean():.2f}%")
    ax2.axvline(_rets.median(),color="#2980b9", lw=1.2, ls="--", label=f"Median {_rets.median():.2f}%")
    mpl_style(ax2, fig2)
    ax2.set_xlabel("Daily Return (%)", fontsize=9)
    ax2.set_ylabel("Frequency",        fontsize=9)
    ax2.legend(fontsize=8, framealpha=0.9, edgecolor="#cccccc", fancybox=False)
    fig2.tight_layout(pad=0.5)
    story.append(fig_to_rl(fig2, w_cm=16, h_cm=4.5))
    story.append(Paragraph("Figure 2 — Distribution of daily percentage returns", ST_CAP))

    story.append(PageBreak())
    story.append(Paragraph("3. Next-Day Price Prediction", ST_H1))
    story.append(hr(C_RED, thickness=1.5, space_after=10))

    if not pred_available:
        story.append(Paragraph(
            "⚠  Prediction unavailable — no model loaded or insufficient data "
            "(minimum 60 rows required for feature engineering).",
            S("NM", fontSize=10, textColor=C_ORANGE)))
    else:
        story.append(section_bar("3.1  Prediction Metrics"))
        story.append(Spacer(1, 0.3 * cm))

        delta_sign  = "+" if change >= 0 else ""
        pred_items  = [
            ("Last Close Price",         f"${last_close:,.2f}",   None),
            ("Predicted Next-Day Close", f"${pred_next:,.2f}",
             f"{delta_sign}{change:.2f} ({delta_sign}{pct_change:.2f}%)"),
            ("Direction",                signal,                   None),
            ("Data as of",               str(data_to),            None),
        ]
        story.append(metric_cards(pred_items, cols=4))
        story.append(Spacer(1, 0.5 * cm))

        story.append(section_bar("3.2  Buy / Sell Signal"))
        story.append(Spacer(1, 0.3 * cm))

        is_buy = signal == "BUY"
        sig_color  = C_GREEN if is_buy else C_RED
        sig_bg     = colors.HexColor("#e8f5e9") if is_buy else colors.HexColor("#fdecea")
        sig_icon   = "🟢  BUY" if is_buy else "🔴  SELL"
        sig_body   = (
            f"The model predicts a next-day close of <b>${pred_next:,.2f}</b>, which is "
            f"<b>${abs(change):.2f} ({abs(pct_change):.2f}%)</b> "
            f"{'above' if is_buy else 'below'} the last close of <b>${last_close:,.2f}</b>. "
            f"This constitutes a <b>{'bullish' if is_buy else 'bearish'}</b> signal."
        )

        sig_tbl = Table(
            [[Paragraph(sig_icon, ST_SIG_BUY if is_buy else ST_SIG_SELL),
              Paragraph(sig_body, ST_BODY)]],
            colWidths=[3.5 * cm, inner_w - 3.5 * cm])
        sig_tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), sig_bg),
            ("GRID",          (0,0), (-1,-1), 0.8,
             C_GREEN if is_buy else C_RED),
            ("TOPPADDING",    (0,0), (-1,-1), 14),
            ("BOTTOMPADDING", (0,0), (-1,-1), 14),
            ("LEFTPADDING",   (0,0), (-1,-1), 12),
            ("RIGHTPADDING",  (0,0), (-1,-1), 12),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ]))
        story.append(sig_tbl)
        story.append(Spacer(1, 0.5 * cm))

        story.append(Paragraph(
            "The pre-trained Linear Regression model uses lag features (Close Lag 1–5) "
            "and rolling statistics (7- and 30-day mean/std) to generate this forecast. "
            "Past performance is not indicative of future results.",
            ST_BODY))

    story.append(PageBreak())
    story.append(Paragraph("4. Recent Trading Data", ST_H1))
    story.append(hr(C_RED, thickness=1.5, space_after=10))
    story.append(section_bar("4.1  Last 10 Trading Days"))
    story.append(Spacer(1, 0.3 * cm))

    _recent = live_df[['Open','High','Low','Close','Volume']].tail(10).copy()
    _recent['Daily Chg %'] = _recent['Close'].pct_change().mul(100)
    _recent = _recent.iloc[::-1]

    rec_headers = ["Date", "Open", "High", "Low", "Close", "Volume", "Chg %"]
    rec_rows = []
    for idx, row in _recent.iterrows():
        chg_val = row['Daily Chg %']
        chg_str = f"{chg_val:+.2f}%" if not np.isnan(chg_val) else "—"
        rec_rows.append([
            str(idx.date()),
            f"${row['Open']:.2f}",
            f"${row['High']:.2f}",
            f"${row['Low']:.2f}",
            f"${row['Close']:.2f}",
            f"{int(row['Volume']):,}",
            chg_str,
        ])
    cws = [inner_w*w for w in [0.14,0.12,0.12,0.12,0.12,0.22,0.16]]

    rec_base_style = [
        ("BACKGROUND",    (0,0),  (-1,0),  C_RED),
        ("TEXTCOLOR",     (0,0),  (-1,0),  colors.white),
        ("FONTNAME",      (0,0),  (-1,0),  "Helvetica-Bold"),
        ("GRID",          (0,0),  (-1,-1), 0.5, C_BORDER),
        ("TOPPADDING",    (0,0),  (-1,-1), 6),
        ("BOTTOMPADDING", (0,0),  (-1,-1), 6),
        ("LEFTPADDING",   (0,0),  (-1,-1), 8),
        ("FONTSIZE",      (0,0),  (-1,-1), 9),
        ("VALIGN",        (0,0),  (-1,-1), "MIDDLE"),
    ]
    for i in range(len(rec_rows)):
        bg = C_ROW_ALT if i % 2 == 0 else C_ROW_MAIN
        rec_base_style.append(("BACKGROUND", (0, i+1), (-1, i+1), bg))

    for i, row_data in enumerate(rec_rows):
        chg_str = row_data[6]  
        try:
            chg_val = float(chg_str.replace("%", "").replace("+", ""))
        except ValueError:
            chg_val = 0.0
        txt_color = C_GREEN if chg_val > 0 else (C_RED if chg_val < 0 else C_MUTED)
        rec_base_style.append(("TEXTCOLOR", (6, i+1), (6, i+1), txt_color))
        rec_base_style.append(("FONTNAME",  (6, i+1), (6, i+1), "Helvetica-Bold"))

    _rec_header_row = [Paragraph(f"<b>{h}</b>",
                                 S("TH3", fontSize=9, textColor=colors.white,
                                   fontName="Helvetica-Bold")) for h in rec_headers]
    _rec_data_rows  = [[Paragraph(str(c), ST_CELL) for c in row] for row in rec_rows]
    rec_tbl = Table([_rec_header_row] + _rec_data_rows, colWidths=cws, repeatRows=1)
    rec_tbl.setStyle(TableStyle(rec_base_style))
    story.append(rec_tbl)
    story.append(Spacer(1, 0.5 * cm))

    story.append(section_bar("5.1  Volume Activity — Last 30 Days"))
    story.append(Spacer(1, 0.25 * cm))

    _vol_data   = live_df.tail(30).copy()
    _vol_rets   = _vol_data['Close'].pct_change().fillna(0)
    _vol_colors = ['#27ae60' if r >= 0 else '#c0392b' for r in _vol_rets]

    fig3, ax3 = plt.subplots(figsize=(13, 3.8))
    bars = ax3.bar(_vol_data.index, _vol_data['Volume'] / 1e6,
                   color=_vol_colors, width=0.7, edgecolor="none")
    mpl_style(ax3, fig3)
    ax3.set_ylabel("Volume (millions)", fontsize=9)
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}M"))
    _buy_patch  = mpatches.Patch(color='#27ae60', label='Up day')
    _sell_patch = mpatches.Patch(color='#c0392b', label='Down day')
    ax3.legend(handles=[_buy_patch, _sell_patch],
               fontsize=8, framealpha=0.9, edgecolor="#cccccc", fancybox=False)
    fig3.tight_layout(pad=0.5)
    story.append(fig_to_rl(fig3, w_cm=16, h_cm=5))
    story.append(Paragraph(
        "Figure 3 — TSLA daily trading volume over the last 30 sessions "
        "(green = price up, red = price down)", ST_CAP))

    story.append(Spacer(1, 1.5 * cm))
    story.append(hr(C_BORDER, thickness=0.8))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(
        f"End of Report  \u2014  StockVision  \u00b7  Tesla (TSLA) Live Analysis  \u00b7  "
        f"Generated {now_str}",
        ST_FOOT))

    doc.build(story, onFirstPage=_draw_hf, onLaterPages=_draw_hf)
    buf.seek(0)
    return buf.read()


df           = None
model_bytes  = None
source_label = ""
load_error   = None
info_msg     = None

nav_tab4, nav_tab0, nav_tab1, nav_tab2, nav_tab3 = st.tabs([
    "⚡ Live Tesla",
    "Manual Stock",
    "Market Overview",
    "Insights",
    "Price Forecasting",
])

with nav_tab0:
    st.markdown("<h2 style='margin-bottom:4px;'>Manual Stock Setup</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8b949e; margin-bottom:20px;'>Configure your prediction model and load stock data before exploring the dashboard.</p>", unsafe_allow_html=True)

    _col_model, _spacer, _col_src = st.columns([5, 1, 6])

    with _col_model:
        st.markdown("""
        <div style='background:#161b22; border:1px solid #30363d; border-radius:10px; padding:18px 20px 8px 20px; margin-bottom:6px;'>
        <div style='font-size:15px; font-weight:700; color:#e6edf3; margin-bottom:12px;'>Prediction Model</div>
        """, unsafe_allow_html=True)
        model_bytes = None

        _auto_pkl = None
        _app_dir = os.path.dirname(os.path.abspath(__file__))
        _pkl_files = [f for f in os.listdir(_app_dir) if f.endswith('.pkl')]
        if _pkl_files:
            _auto_pkl = os.path.join(_app_dir, _pkl_files[0])

        _uploaded_model = st.file_uploader(
            "Upload model (.pkl)",
            type=["pkl"],
            key="model_upload",
            help="Upload a pre-trained scikit-learn Linear Regression model (.pkl)"
        )

        if _uploaded_model is not None:
            try:
                model_bytes = _uploaded_model.getvalue()
                _test_model = pickle.loads(model_bytes)
                st.markdown(
                    f"<div class='success-box'>✔ Model loaded from upload — "
                    f"{_test_model.n_features_in_} features</div>",
                    unsafe_allow_html=True
                )
            except Exception as _e:
                st.markdown(
                    f"<div class='warn-box'>⚠ Could not load uploaded model: {_e}</div>",
                    unsafe_allow_html=True
                )
                model_bytes = None
        elif _auto_pkl:
            try:
                with open(_auto_pkl, "rb") as _f:
                    model_bytes = _f.read()
                _test_model = pickle.loads(model_bytes)
                st.markdown(
                    f"<div class='success-box'>✔ Model auto-loaded: <code>{os.path.basename(_auto_pkl)}</code> — "
                    f"{_test_model.n_features_in_} features</div>",
                    unsafe_allow_html=True
                )
            except Exception as _e:
                st.markdown(
                    f"<div class='warn-box'>⚠ Could not load model: {_e}</div>",
                    unsafe_allow_html=True
                )
                model_bytes = None
        else:
            st.markdown(
                "<div class='warn-box'>⚠ No model loaded. Upload a <strong>.pkl</strong> file above "
                "to enable the Price Forecasting tab.</div>",
                unsafe_allow_html=True
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with _col_src:
        st.markdown("""
        <div style='background:#161b22; border:1px solid #30363d; border-radius:10px; padding:18px 20px 8px 20px; margin-bottom:6px;'>
        <div style='font-size:15px; font-weight:700; color:#e6edf3; margin-bottom:12px;'>Data Source</div>
        """, unsafe_allow_html=True)

        dl_c1, dl_c2 = st.columns([1, 1])
        sample_fmt = dl_c1.selectbox(
            "Download sample file",
            ["CSV", "JSON", "SQL"],
            key="sample_fmt",
        )
        _sample_map = {
            "CSV":  (generate_sample_csv,  "sample_stock_data.csv",  "text/csv"),
            "JSON": (generate_sample_json, "sample_stock_data.json", "application/json"),
            "SQL":  (generate_sample_sql,  "sample_stock_data.sql",  "text/plain"),
        }
        _fn, _fname, _mime = _sample_map[sample_fmt]
        dl_c2.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        dl_c2.download_button(
            label="⬇ Download Sample",
            data=_fn(),
            file_name=_fname,
            mime=_mime,
            use_container_width=True,
            key="dl_sample_unified"
        )

        st.markdown("<hr style='border-color:#21262d; margin:6px 0 10px 0;'>", unsafe_allow_html=True)

        src_c1, src_c2 = st.columns([1, 2])
        source = src_c1.radio(
            "Select data source",
            ["CSV", "JSON", "SQL Database"],
            label_visibility="collapsed"
        )

        with src_c2:
            if source == "CSV":
                source_label = "CSV"
                csv_files = st.file_uploader(
                    "Upload one or more CSV files",
                    type=["csv"],
                    accept_multiple_files=True,
                    key="csv_upload"
                )
                if csv_files:
                    try:
                        frames = []
                        for f in csv_files:
                            part = _load_csv(f.getvalue())
                            frames.append(part)
                        if len(frames) > 1:
                            raw_df = pd.concat(frames, ignore_index=True)
                            info_msg = f"{len(frames)} files merged ({len(raw_df):,} total rows)"
                        else:
                            raw_df = frames[0]
                        df, msg = validate_and_prepare(raw_df)
                        if df is None:
                            load_error = msg
                        elif msg:
                            info_msg = (info_msg + " · " + msg) if info_msg else msg
                        if df is not None:
                            _name = os.path.splitext(csv_files[0].name)[0].upper().replace('_', ' ').replace('-', ' ')
                            st.session_state['ticker_name'] = _name
                    except Exception as e:
                        load_error = f"Failed to read CSV(s): {e}"

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
                        if df is not None:
                            _name = os.path.splitext(json_file.name)[0].upper().replace('_', ' ').replace('-', ' ')
                            st.session_state['ticker_name'] = _name
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
                                if df is not None:
                                    _name = selected_table.upper().replace('_', ' ').replace('-', ' ')
                                    st.session_state['ticker_name'] = _name
                    except Exception as e:
                        load_error = f"Failed to read SQL file: {e}"

        if load_error:
            st.markdown(f"<div class='warn-box'>{load_error}</div>", unsafe_allow_html=True)
        elif df is not None:
            status_c1, status_c2 = st.columns([2, 1])
            status_c1.markdown(
                f"<div class='success-box'>{len(df):,} rows loaded "
                f"<span class='source-badge'>{source_label}</span></div>",
                unsafe_allow_html=True
            )
            if info_msg:
                st.markdown(f"<div class='warn-box'>{info_msg}</div>", unsafe_allow_html=True)
            with status_c2:
                with st.spinner("Generating PDF report…"):
                    _ticker = st.session_state.get('ticker_name', 'Stock')
                    _pdf_bytes = generate_report_pdf(df, model_bytes, ticker_name=_ticker)
                _safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', _ticker).lower()
                st.download_button(
                    label="Download Report PDF",
                    data=_pdf_bytes,
                    file_name=f"stockvision_{_safe_name}_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="dl_report_pdf"
                )
        st.markdown("</div>", unsafe_allow_html=True)

    if df is not None:
        st.markdown("""
        <div class='success-box' style='margin-top:16px; font-size:14px;'>
        ✅ <strong>Setup complete!</strong> Switch to the <strong>Market Overview</strong>,
        <strong>Insights</strong>, or <strong>Price Forecasting</strong> tabs to explore your data.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='info-box' style='margin-top:16px;'>
        <b>Required columns in your dataset:</b><br>
        <code>Date, Open, High, Low, Close, Volume</code>
        </div>
        """, unsafe_allow_html=True)

with nav_tab4:
    st.markdown("<h1>⚡ Live Tesla (TSLA) — Real-Time Prediction</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#8b949e;'>Fetches live TSLA data via <strong>yfinance</strong>, "
        "engineers features, and predicts the next-day closing price using your loaded model. "
        "Data is cached for <strong>5 minutes</strong>.</p>",
        unsafe_allow_html=True,
    )

    if not _YF_AVAILABLE:
        st.error("yfinance is not installed. Run `pip install yfinance` and restart the app.")
        st.stop()

    _period_default  = st.session_state.get("live_period_tab",  "5y")
    _interval_default = st.session_state.get("live_intv_tab",   "1d")

    with st.spinner("Loading live TSLA data…"):
        live_df, live_err = fetch_live_tsla_data(_period_default, _interval_default)

    if live_df is None:
        st.markdown(
            f"<div class='warn-box'>⚠ Could not load live data: {live_err}</div>",
            unsafe_allow_html=True,
        )
        st.stop()

    if live_err:
        st.markdown(f"<div class='warn-box'>ℹ {live_err}</div>", unsafe_allow_html=True)

    ctrl_c1, ctrl_c2, ctrl_c3, ctrl_c4 = st.columns([2, 2, 2, 2])
    live_period   = ctrl_c1.selectbox("Data Period", ["1mo","3mo","6mo","1y","2y","5y","max"],
                                      index=["1mo","3mo","6mo","1y","2y","5y","max"].index(_period_default) if _period_default in ["1mo","3mo","6mo","1y","2y","5y","max"] else 5,
                                      key="live_period_tab")
    live_interval = ctrl_c2.selectbox("Interval",   ["1d","1h","5m"],
                                      index=["1d","1h","5m"].index(_interval_default),
                                      key="live_intv_tab")

    with ctrl_c3:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        with st.spinner(""):
            _live_pdf_bytes_top = generate_live_tesla_pdf(
                live_df, model_bytes,
                live_period=live_period,
                live_interval=live_interval,
            )
        st.download_button(
            label="⬇ Download Live Tesla PDF",
            data=_live_pdf_bytes_top,
            file_name="stockvision_tesla_live_report.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="dl_live_tesla_pdf_top",
        )

    with ctrl_c4:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        refresh_btn = st.button("🔄 Refresh Live Data", key="live_refresh", use_container_width=True)

    if refresh_btn:
        st.cache_data.clear()
        st.rerun()

    if live_period != _period_default or live_interval != _interval_default:
        with st.spinner("Reloading with new settings…"):
            live_df, live_err = fetch_live_tsla_data(live_period, live_interval)
        if live_df is None:
            st.markdown(
                f"<div class='warn-box'>⚠ Could not load live data: {live_err}</div>",
                unsafe_allow_html=True,
            )
            st.stop()

    _last_close_live  = float(live_df['Close'].iloc[-1])
    _prev_close_live  = float(live_df['Close'].iloc[-2]) if len(live_df) > 1 else _last_close_live
    _day_chg          = _last_close_live - _prev_close_live
    _day_pct          = (_day_chg / _prev_close_live) * 100 if _prev_close_live else 0
    _52w_high         = float(live_df['Close'].max())
    _52w_low          = float(live_df['Close'].min())
    _avg_vol          = live_df['Volume'].mean()

    st.markdown("<div class='section-header'>Live Market Snapshot</div>", unsafe_allow_html=True)
    lk1, lk2, lk3, lk4, lk5 = st.columns(5)
    lk1.metric("Latest Close",     f"${_last_close_live:,.2f}",
               delta=f"{_day_chg:+.2f} ({_day_pct:+.2f}%)")
    lk2.metric("Period High",      f"${_52w_high:,.2f}")
    lk3.metric("Period Low",       f"${_52w_low:,.2f}")
    lk4.metric("Avg Daily Volume", f"{int(_avg_vol/1e6):.1f}M")
    lk5.metric("Data Points",      f"{len(live_df):,}")

    st.markdown("<div class='section-header'>Tesla Live Price Trend</div>", unsafe_allow_html=True)

    fig_live = go.Figure()

    fig_live.add_trace(go.Scatter(
        x=live_df.index, y=live_df['Close'],
        fill='tozeroy',
        fillcolor='rgba(232,39,59,0.08)',
        line=dict(color='#e8273b', width=2),
        name='Close Price',
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Close: $%{y:,.2f}<extra></extra>',
    ))

    _ma20_live = live_df['Close'].rolling(20).mean()
    fig_live.add_trace(go.Scatter(
        x=live_df.index, y=_ma20_live,
        line=dict(color='#f0a500', width=1.2, dash='dot'),
        name='MA 20',
        hovertemplate='MA20: $%{y:,.2f}<extra></extra>',
    ))

    fig_live.update_layout(
        **PLOTLY_LAYOUT,
        title='Tesla Live Price Trend',
        xaxis_title='Date',
        yaxis_title='Close Price ($)',
        height=420,
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        hovermode='x unified',
    )
    st.plotly_chart(fig_live, use_container_width=True)

    st.markdown("<div class='section-header'>Next-Day Price Prediction</div>", unsafe_allow_html=True)

    if model_bytes is None:
        st.markdown(
            "<div class='warn-box'>⚠ No model loaded. Upload a <strong>.pkl</strong> model "
            "in the <strong>Manual Stock</strong> tab to enable live predictions.</div>",
            unsafe_allow_html=True,
        )
    else:
        try:
            loaded_model_live  = pickle.loads(model_bytes)
            required_feats_live = list(loaded_model_live.feature_names_in_)

            if len(live_df) < 60:
                st.markdown(
                    "<div class='warn-box'>⚠ Insufficient live data rows for feature engineering "
                    "(need ≥ 60). Try a longer period.</div>",
                    unsafe_allow_html=True,
                )
            else:
                live_model_data = engineer_features(live_df)

                missing_feats_live = [f for f in required_feats_live if f not in live_model_data.columns]
                if missing_feats_live:
                    st.error(f"Feature mismatch — live data is missing: `{missing_feats_live}`")
                else:
                    feature_cols_live = required_feats_live
                    latest_row = live_model_data[feature_cols_live].iloc[-1].copy()
                    pred_next  = loaded_model_live.predict(latest_row.values.reshape(1, -1))[0]

                    live_change     = pred_next - _last_close_live
                    live_pct_change = (live_change / _last_close_live) * 100

                    pm1, pm2, pm3 = st.columns(3)
                    pm1.metric("Last Close Price",          f"${_last_close_live:,.2f}")
                    pm2.metric("Predicted Next-Day Close",  f"${pred_next:,.2f}",
                               delta=f"{live_change:+.2f} ({live_pct_change:+.2f}%)")
                    pm3.metric("Data as of",
                               str(live_df.index[-1].date()))

                    _is_buy = pred_next > _last_close_live
                    _signal_color  = "#3fb950" if _is_buy else "#e8273b"
                    _signal_bg     = "#0d2818" if _is_buy else "#2d0a0a"
                    _signal_border = "#3fb950" if _is_buy else "#e8273b"
                    _signal_icon   = "🟢" if _is_buy else "🔴"
                    _signal_label  = "BUY" if _is_buy else "SELL"
                    _signal_desc   = (
                        f"Predicted price <strong>${pred_next:,.2f}</strong> is "
                        f"<strong>above</strong> last close <strong>${_last_close_live:,.2f}</strong> — "
                        f"bullish signal."
                        if _is_buy else
                        f"Predicted price <strong>${pred_next:,.2f}</strong> is "
                        f"<strong>below</strong> last close <strong>${_last_close_live:,.2f}</strong> — "
                        f"bearish signal."
                    )

                    st.markdown(f"""
                    <div style="
                        background:{_signal_bg};
                        border-left: 5px solid {_signal_border};
                        border-radius: 10px;
                        padding: 20px 24px;
                        margin: 16px 0;
                        display: flex;
                        align-items: center;
                        gap: 20px;
                    ">
                        <div style="font-size:3rem; line-height:1;">{_signal_icon}</div>
                        <div>
                            <div style="font-size:1.6rem; font-weight:800;
                                        color:{_signal_color}; letter-spacing:2px;">
                                {_signal_label} SIGNAL
                            </div>
                            <div style="color:#c9d1d9; font-size:13px; margin-top:6px;">
                                {_signal_desc}
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown(f"""
                    <div class='info-box'>
                    <strong>Live Prediction Summary:</strong> Based on the pre-trained Linear Regression
                    model with lag features and rolling statistics, the forecast for the next trading
                    session's closing price is <strong>${pred_next:,.2f}</strong> — a change of
                    <strong>{live_change:+.2f} ({live_pct_change:+.2f}%)</strong> from the
                    most recent close of <strong>${_last_close_live:,.2f}</strong>.
                    <br><br>
                    <em>⚠ This is a model output for demonstration purposes and should not be
                    treated as financial advice.</em>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("<div class='section-header'>Recent TSLA Data (Last 10 Trading Days)</div>", unsafe_allow_html=True)
                    _recent = live_df[['Open','High','Low','Close','Volume']].tail(10).copy()
                    _recent['Daily Chg %'] = _recent['Close'].pct_change().mul(100).round(2)
                    _recent = _recent.iloc[::-1]
                    # Reset index to integers to avoid non-unique index error with Styler,
                    # then display date as a formatted column instead.
                    _recent = _recent.reset_index()
                    date_col = _recent.columns[0] 
                    _recent[date_col] = pd.to_datetime(_recent[date_col]).dt.date
                    _recent = _recent.rename(columns={date_col: 'Date'})
                    st.dataframe(
                        _recent.style
                            .format({
                                'Open':       '${:.2f}',
                                'High':       '${:.2f}',
                                'Low':        '${:.2f}',
                                'Close':      '${:.2f}',
                                'Volume':     '{:,.0f}',
                                'Daily Chg %':'  {:.2f}%',
                            })
                            .map(
                                lambda v: 'color: #3fb950' if isinstance(v, float) and v > 0
                                          else ('color: #e8273b' if isinstance(v, float) and v < 0 else ''),
                                subset=['Daily Chg %']
                            ),
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.markdown("<div class='section-header'>Recent Volume Activity</div>", unsafe_allow_html=True)
                    _vol_data = live_df.tail(30).copy()
                    _vol_colors = [
                        '#3fb950' if r > 0 else '#e8273b'
                        for r in _vol_data['Close'].pct_change().fillna(0)
                    ]
                    fig_vol = go.Figure(go.Bar(
                        x=_vol_data.index,
                        y=_vol_data['Volume'],
                        marker_color=_vol_colors,
                        name='Volume',
                        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Volume: %{y:,.0f}<extra></extra>',
                    ))
                    fig_vol.update_layout(
                        **PLOTLY_LAYOUT,
                        title='TSLA Daily Volume (Last 30 Days) — Green = Up Day, Red = Down Day',
                        xaxis_title='Date',
                        yaxis_title='Volume',
                        height=320,
                    )
                    st.plotly_chart(fig_vol, use_container_width=True)

                    _fetched_at = datetime.datetime.now().strftime("%H:%M:%S")
                    st.markdown(
                        f"<div style='color:#8b949e; font-size:11px; text-align:right; margin-top:8px;'>"
                        f"🕒 Data cached for 5 min · Last fetched at {_fetched_at} · "
                        f"Press <strong>Refresh Live Data</strong> to reload early.</div>",
                        unsafe_allow_html=True,
                    )

        except Exception as _live_exc:
            st.error(f"Live prediction error: {_live_exc}")
            import traceback
            st.code(traceback.format_exc())

if df is None:
    for _tab in [nav_tab1, nav_tab2, nav_tab3]:
        with _tab:
            st.info("⚙️ Configure your data and model in the **Manual Stock** tab to get started.")

if df is None:
    st.stop()


with nav_tab1:
    _ticker = st.session_state.get('ticker_name', 'Stock')
    st.markdown(f"<h1>StockVision — {_ticker} Market Overview</h1>", unsafe_allow_html=True)
    st.markdown(
        f"<p>Comprehensive analysis of <strong>{_ticker}</strong> — "
        f"{df.index.min().date()} to {df.index.max().date()}</p>",
        unsafe_allow_html=True
    )

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
    fig_price.update_layout(**PLOTLY_LAYOUT, title=f'{_ticker} Closing Price with Moving Averages',
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
        ("Total Return",   f"Delivered a total return of <strong>{rm['Total Return (%)']:,.1f}%</strong> across the dataset period."),
        ("Volatility",     f"Annualized volatility of <strong>{rm['Annualized Volatility (%)']:.1f}%</strong> — {'high' if rm['Annualized Volatility (%)'] > 30 else 'moderate'}-risk profile."),
        ("Sharpe Ratio",   f"A Sharpe ratio of <strong>{rm['Sharpe Ratio']}</strong> indicates returns reasonably compensate for risk taken."),
        ("Drawdown",       f"Maximum drawdown of <strong>{rm['Max Drawdown (%)']:.1f}%</strong> highlights downside exposure during correction periods."),
        ("Positive Days",  f"Closed higher on <strong>{rm['Positive Days (%)']:.1f}%</strong> of all trading days in the dataset."),
    ]
    for title, body in insights:
        st.markdown(f"<div class='insight-box'><strong>{title}:</strong> {body}</div>", unsafe_allow_html=True)

with nav_tab2:
    _ticker = st.session_state.get('ticker_name', 'Stock')
    st.markdown("<h1>Exploratory Data Analysis</h1>", unsafe_allow_html=True)
    st.markdown(f"<p>Deep dive into <strong>{_ticker}</strong> stock data — distributions, correlations, volatility and time series patterns.</p>", unsafe_allow_html=True)

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
            'dtype':   df.dtypes.astype(str).values,
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

with nav_tab3:
    _ticker = st.session_state.get('ticker_name', 'Stock')
    st.markdown("<h1>Linear Regression — Price Prediction</h1>", unsafe_allow_html=True)
    st.markdown(f"<p>Evaluate the pre-trained Linear Regression model and explore next-day price forecasts for <strong>{_ticker}</strong>.</p>", unsafe_allow_html=True)

    if model_bytes is None:
        st.warning(
            "**No model loaded.** Upload a `.pkl` model file using the "
            "**Manual Stock** panel above to enable predictions."
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
    fig_avp.update_layout(**PLOTLY_LAYOUT, title=f'Linear Regression: Actual vs Predicted Close Price — {_ticker}',
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