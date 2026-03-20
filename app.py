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
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
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
        font-size: 15px !important;
        font-weight: 500 !important;
        padding: 10px 22px !important;
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

# ── Centered App Title ──────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 28px 0 8px 0;">
    <span style="font-size:2.8rem; font-weight:800; letter-spacing:1px;
                 background: linear-gradient(90deg,#e8273b,#ff6b6b);
                 -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
        StockVision
    </span>
    <div style="color:#8b949e; font-size:1rem; margin-top:4px; letter-spacing:2px; text-transform:uppercase;">
        Tesla (TSLA) · Stock Analysis &amp; Prediction
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
    dates  = pd.date_range(end=datetime.date.today(), periods=120, freq='B')
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
        "-- Sample Tesla stock data — import with the SQL Database source",
        "CREATE TABLE IF NOT EXISTS tsla_stock (",
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
            f"INSERT INTO tsla_stock VALUES "
            f"('{row['Date']}', {row['Open']}, {row['High']}, "
            f"{row['Low']}, {row['Close']}, {int(row['Volume'])});"
        )
    return "\n".join(lines).encode('utf-8')



def generate_report_pdf(df, model_bytes):
    """
    Generate a polished, professional light-theme PDF report.
    Covers Dashboard (key metrics, price history, drawdown, risk, insights)
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
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate

    # ── Palette (light / professional) ──────────────────────────────────────
    C_RED       = colors.HexColor("#c0392b")      # accent
    C_RED_LIGHT = colors.HexColor("#f8d7da")      # tinted bg
    C_DARK      = colors.HexColor("#1a1a2e")      # headings
    C_BODY      = colors.HexColor("#2c2c2c")      # body text
    C_MUTED     = colors.HexColor("#666666")      # captions / labels
    C_BORDER    = colors.HexColor("#dee2e6")      # table borders
    C_ROW_ALT   = colors.HexColor("#f8f9fa")      # alternating row
    C_ROW_MAIN  = colors.white
    C_HDR_BG    = colors.HexColor("#c0392b")      # table header bg
    C_HDR_TEXT  = colors.white
    C_GREEN     = colors.HexColor("#27ae60")
    C_ORANGE    = colors.HexColor("#e67e22")
    C_BLUE      = colors.HexColor("#2980b9")
    C_CARD_BG   = colors.HexColor("#f0f4ff")      # metric card bg

    W, H = A4
    L_MAR = R_MAR = 1.8*cm
    T_MAR = 3.2*cm   # leave room for header band
    B_MAR = 2.2*cm   # leave room for footer
    inner_w = W - L_MAR - R_MAR
    now_str = datetime.datetime.now().strftime("%d %b %Y  %H:%M")

    # ── Page header / footer via canvas callbacks ────────────────────────────
    def _draw_header_footer(canv, doc):
        canv.saveState()
        # ── top band ──
        canv.setFillColor(C_RED)
        canv.rect(0, H - 1.8*cm, W, 1.8*cm, fill=1, stroke=0)
        canv.setFillColor(colors.white)
        canv.setFont("Helvetica-Bold", 13)
        canv.drawString(L_MAR, H - 1.15*cm, "StockVision")
        canv.setFont("Helvetica", 8)
        canv.drawRightString(W - R_MAR, H - 0.85*cm, "Tesla (TSLA)  |  Stock Analysis Report")
        canv.drawRightString(W - R_MAR, H - 1.35*cm, now_str)
        # ── bottom footer ──
        canv.setFillColor(C_MUTED)
        canv.setFont("Helvetica", 7)
        canv.drawString(L_MAR, 1.0*cm,
                        "Generated by StockVision  |  For informational purposes only — not investment advice.")
        canv.drawRightString(W - R_MAR, 1.0*cm, f"Page {doc.page}")
        canv.setStrokeColor(C_BORDER)
        canv.setLineWidth(0.5)
        canv.line(L_MAR, 1.5*cm, W - R_MAR, 1.5*cm)
        canv.restoreState()

    buf = _io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=L_MAR, rightMargin=R_MAR,
        topMargin=T_MAR, bottomMargin=B_MAR,
        onPage=_draw_header_footer,
    )

    # ── Text styles ──────────────────────────────────────────────────────────
    ss = getSampleStyleSheet()

    def S(name, parent="Normal", **kw):
        return ParagraphStyle(name, parent=ss[parent], **kw)

    ST_COVER_TITLE = S("CTitle", fontSize=36, textColor=C_RED,
                       fontName="Helvetica-Bold", alignment=TA_CENTER, spaceAfter=6)
    ST_COVER_SUB   = S("CSub",   fontSize=13, textColor=C_MUTED,
                       alignment=TA_CENTER, spaceAfter=4)
    ST_COVER_BODY  = S("CBody",  fontSize=10, textColor=C_BODY,
                       alignment=TA_CENTER, leading=16, spaceAfter=4)

    ST_H1    = S("H1",    fontSize=17, textColor=C_DARK, fontName="Helvetica-Bold",
                 spaceBefore=14, spaceAfter=4)
    ST_H2    = S("H2",    fontSize=12, textColor=C_RED,  fontName="Helvetica-Bold",
                 spaceBefore=10, spaceAfter=3)
    ST_BODY  = S("Body",  fontSize=9,  textColor=C_BODY, leading=14)
    ST_CAP   = S("Cap",   fontSize=8,  textColor=C_MUTED, alignment=TA_CENTER,
                 spaceBefore=3, spaceAfter=8, fontName="Helvetica-Oblique")
    ST_CELL  = S("Cell",  fontSize=9,  textColor=C_BODY)
    ST_CELLB = S("CellB", fontSize=9,  textColor=C_BODY, fontName="Helvetica-Bold")
    ST_LBL   = S("Lbl",   fontSize=8,  textColor=C_MUTED, alignment=TA_CENTER)
    ST_VAL   = S("Val",   fontSize=18, textColor=C_DARK, fontName="Helvetica-Bold",
                 alignment=TA_CENTER)
    ST_DELTA_G = S("DG",  fontSize=8, textColor=C_GREEN, alignment=TA_CENTER)
    ST_DELTA_R = S("DR",  fontSize=8, textColor=colors.HexColor("#c0392b"), alignment=TA_CENTER)
    ST_FOOT  = S("Foot",  fontSize=7, textColor=C_MUTED, alignment=TA_CENTER)

    # ── Helpers ──────────────────────────────────────────────────────────────
    def hr(color=C_BORDER, thickness=0.8, space_before=0, space_after=8):
        return HRFlowable(width="100%", thickness=thickness, color=color,
                          spaceBefore=space_before, spaceAfter=space_after)

    def section_bar(title):
        """Red-background section title bar."""
        tbl = Table([[Paragraph(f'<font color="white"><b>{title}</b></font>',
                                S("SB", fontSize=11, textColor=colors.white,
                                  fontName="Helvetica-Bold"))]],
                    colWidths=[inner_w])
        tbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), C_RED),
            ("TOPPADDING",    (0,0), (-1,-1), 7),
            ("BOTTOMPADDING", (0,0), (-1,-1), 7),
            ("LEFTPADDING",   (0,0), (-1,-1), 12),
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
        val_row, lbl_row = [], []
        for label, value, delta in items:
            if delta:
                is_pos = not delta.startswith("-")
                dp = Paragraph(delta, ST_DELTA_G if is_pos else ST_DELTA_R)
            else:
                dp = Paragraph("", ST_LBL)
            val_row.append([Paragraph(value, ST_VAL), dp])
            lbl_row.append([Paragraph(label, ST_LBL)])
        t = Table([val_row, lbl_row], colWidths=[cw]*cols)
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), C_CARD_BG),
            ("GRID",          (0,0), (-1,-1), 0.6, C_BORDER),
            ("TOPPADDING",    (0,0), (-1,-1), 10),
            ("BOTTOMPADDING", (0,0), (-1,-1), 10),
            ("LEFTPADDING",   (0,0), (-1,-1), 6),
            ("RIGHTPADDING",  (0,0), (-1,-1), 6),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
            ("LINEBELOW",     (0,0), (-1,0),  0.5, C_BORDER),
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

    # ── Pre-compute ─────────────────────────────────────────────────────────
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

    # ════════════════════════════════════════════════════════════════════════
    # PAGE 1 — COVER
    # ════════════════════════════════════════════════════════════════════════
    story.append(Spacer(1, 2.5*cm))
    story.append(Paragraph("StockVision", ST_COVER_TITLE))
    story.append(Paragraph("Tesla Inc. (TSLA) &mdash; Stock Analysis &amp; Prediction Report",
                            ST_COVER_SUB))
    story.append(Spacer(1, 0.3*cm))
    story.append(hr(C_RED, thickness=2, space_after=20))

    # Info table
    cover_rows = [
        ["Report Date",       now_str],
        ["Dataset Range",     f"{df.index.min().date()}  \u2192  {df.index.max().date()}"],
        ["Total Trading Days",f"{len(df):,}"],
        ["Model",             "Linear Regression (pre-trained .pkl)"],
        ["Sections",          "Dashboard  |  Risk Analysis  |  Model Prediction"],
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
        "It provides a comprehensive analysis of Tesla Inc. (TSLA) stock data, "
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

    # ════════════════════════════════════════════════════════════════════════
    # PAGE 2 — DASHBOARD: KEY METRICS
    # ════════════════════════════════════════════════════════════════════════
    story.append(Paragraph("1. Dashboard", ST_H1))
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

    # ── Price history chart ─────────────────────────────────────────────────
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
        "Figure 1 — Tesla closing price with 20-, 50-, and 200-day simple moving averages (full dataset)",
        ST_CAP))
    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════════════════
    # PAGE 3 — DRAWDOWN + RISK TABLE + INSIGHTS
    # ════════════════════════════════════════════════════════════════════════
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
    risk_rows_data = [[k, f"{v}{'%' if '%' in k else ''}"] for k, v in rm.items()]
    # clean up double-% from values that already embed %
    risk_rows_data = [[row[0], str(v) if '%' not in str(rm.get(row[0], '')) else str(v)] for row, v in
                      [([k], v) for k, v in rm.items()]]
    risk_rows_clean = [[k, str(v)] for k, v in rm.items()]
    story.append(data_table(risk_headers, risk_rows_clean,
                            col_widths=[inner_w*0.55, inner_w*0.45]))
    story.append(Spacer(1, 0.5*cm))

    story.append(section_bar("1.5  Key Insights"))
    story.append(Spacer(1, 0.25*cm))

    insights = [
        ("Total Return",
         f"Tesla delivered a total return of <b>{rm['Total Return (%)']:,.1f}%</b> "
         f"across the full dataset period."),
        ("Volatility",
         f"Annualized volatility of <b>{rm['Annualized Volatility (%)']:.1f}%</b> "
         f"classifies Tesla as a high-risk, high-reward growth stock."),
        ("Sharpe Ratio",
         f"A Sharpe ratio of <b>{rm['Sharpe Ratio']:.3f}</b> indicates that returns "
         f"reasonably compensate for the risk taken."),
        ("Max Drawdown",
         f"Maximum drawdown of <b>{rm['Max Drawdown (%)']:.1f}%</b> highlights severe "
         f"downside exposure during correction periods."),
        ("Positive Days",
         f"Tesla closed higher on <b>{rm['Positive Days (%)']:.1f}%</b> of all trading days "
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
            "TSLA_linear_model.pkl was not loaded.",
            S("NoModel", fontSize=9, textColor=C_ORANGE)))
        doc.build(story)
        buf.seek(0)
        return buf.read()

    story.append(PageBreak())

    # ════════════════════════════════════════════════════════════════════════
    # PAGE 4 — PREDICTION: MODEL OVERVIEW + ACTUAL VS PREDICTED
    # ════════════════════════════════════════════════════════════════════════
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

    # ════════════════════════════════════════════════════════════════════════
    # PAGE 5 — TRAIN/TEST SPLIT + FORECAST
    # ════════════════════════════════════════════════════════════════════════
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
        f"End of Report  \u2014  StockVision  \u00b7  Tesla (TSLA) Analysis  \u00b7  "
        f"Generated {now_str}",
        ST_FOOT))

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ── Safety defaults ─────────────────────────────────────────────────────────
df           = None
model_bytes  = None
source_label = ""
load_error   = None
info_msg     = None

# ── Data Source & Model — inline expander ────────────────────────────────────
with st.expander("⚙️  Data Source & Model", expanded=True):
    _col_model, _col_src = st.columns([1, 2])

    with _col_model:
        st.markdown("##### 🤖 Prediction Model")
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

    with _col_src:
        st.markdown("##### 📂 Data Source")

        # ── Download sample file (dropdown) ──────────────────────────────
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
            ["CSV", "JSON", "SQL Database", "Google Drive"],
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
                            st.session_state['drive_df']   = df
                            st.session_state['drive_err']  = load_error
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

        # ── Status messages ───────────────────────────────────────────────
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
                    _pdf_bytes = generate_report_pdf(df, model_bytes)
                st.download_button(
                    label="📄 Download Report PDF",
                    data=_pdf_bytes,
                    file_name="stockvision_tesla_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="dl_report_pdf"
                )

st.markdown("<hr style='border-color:#21262d; margin:0 0 4px 0;'>", unsafe_allow_html=True)

# ── Tab navigation ────────────────────────────────────────────────────────────
nav_tab1, nav_tab2, nav_tab3 = st.tabs(["📊  Dashboard", "🔍  EDA", "🤖  Prediction"])

# ── No data loaded — show instructions inside each tab ───────────────────────
if df is None:
    for _tab in [nav_tab1, nav_tab2, nav_tab3]:
        with _tab:
            st.info("Upload your data using the **⚙️ Data Source & Model** panel above to get started.")
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

# ── Dashboard tab ─────────────────────────────────────────────────────────────
with nav_tab1:
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
    st.plotly_chart(fig_price, width="stretch")

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
    st.plotly_chart(fig_candle, width="stretch")

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

with nav_tab2:
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
        st.dataframe(df.head(10), width="stretch")

        st.markdown("**Descriptive Statistics**")
        st.dataframe(df.describe().round(4), width="stretch")

        st.markdown("**Data Types & Missing Values**")
        dtype_df = pd.DataFrame({
            'Column':  df.columns,
            'dtype':   df.dtypes.astype(str).values,
            'Missing': df.isnull().sum().values,
            'Unique':  df.nunique().values,
        })
        st.dataframe(dtype_df, width="stretch", hide_index=True)

    with tab2:
        st.markdown("<div class='section-header'>Price Distribution Analysis</div>", unsafe_allow_html=True)
        price_col = st.selectbox("Select column", ['Close', 'Open', 'High', 'Low'])

        col_a, col_b = st.columns(2)
        with col_a:
            fig_hist = px.histogram(df, x=price_col, nbins=80,
                title=f'{price_col} Price Distribution',
                color_discrete_sequence=['#e8273b'])
            fig_hist.update_layout(**PLOTLY_LAYOUT, height=350)
            st.plotly_chart(fig_hist, width="stretch")

        with col_b:
            fig_box = go.Figure()
            for col_name, color in zip(['Open', 'High', 'Low', 'Close'],
                                        ['#58a6ff','#3fb950','#f0a500','#e8273b']):
                fig_box.add_trace(go.Box(y=df[col_name], name=col_name,
                    marker_color=color, boxmean=True))
            fig_box.update_layout(**PLOTLY_LAYOUT, title='OHLC Price Box Plots', height=350)
            st.plotly_chart(fig_box, width="stretch")

        st.markdown("<div class='section-header'>Yearly Average Close Price</div>", unsafe_allow_html=True)
        yearly = df['Close'].resample('YE').mean().reset_index()
        yearly.columns = ['Year', 'Avg Close']
        yearly['Year'] = yearly['Year'].dt.year
        fig_yr = px.bar(yearly, x='Year', y='Avg Close',
            color='Avg Close', color_continuous_scale='Reds',
            title='Average Closing Price by Year')
        fig_yr.update_layout(**PLOTLY_LAYOUT, height=380)
        st.plotly_chart(fig_yr, width="stretch")

    
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
        st.plotly_chart(fig_rvol, width="stretch")

        st.markdown("<div class='section-header'>Drawdown Analysis</div>", unsafe_allow_html=True)
        roll_max = df['Close'].cummax()
        drawdown = (df['Close'] - roll_max) / roll_max * 100
        fig_dd = go.Figure()
        fig_dd.add_trace(go.Scatter(x=drawdown.index, y=drawdown,
            fill='tozeroy', fillcolor='rgba(232,39,59,0.2)',
            line=dict(color='#e8273b', width=1.5), name='Drawdown'))
        fig_dd.update_layout(**PLOTLY_LAYOUT, title='Drawdown from All-Time High (%)',
                              xaxis_title='Date', yaxis_title='Drawdown (%)', height=380)
        st.plotly_chart(fig_dd, width="stretch")

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
        st.plotly_chart(fig_sharpe, width="stretch")

    with tab4:
        st.markdown("<div class='section-header'>Correlation Matrix</div>", unsafe_allow_html=True)
        corr = df.corr()
        fig_corr = px.imshow(corr, text_auto='.3f', color_continuous_scale='RdBu_r',
            title='Feature Correlation Matrix', aspect='auto')
        fig_corr.update_layout(**PLOTLY_LAYOUT, height=480)
        st.plotly_chart(fig_corr, width="stretch")

        st.markdown("<div class='section-header'>Volume vs Close Price</div>", unsafe_allow_html=True)
        fig_scatter = px.scatter(df.reset_index(), x='Volume', y='Close',
            color='Close', color_continuous_scale='Reds',
            title='Volume vs. Close Price', opacity=0.4)
        fig_scatter.update_layout(**PLOTLY_LAYOUT, height=400)
        st.plotly_chart(fig_scatter, width="stretch")

        st.markdown("<div class='section-header'>Feature Scatter</div>", unsafe_allow_html=True)
        c_x, c_y = st.columns(2)
        feat_x = c_x.selectbox("X-axis", df.columns.tolist(), index=0)
        feat_y = c_y.selectbox("Y-axis", df.columns.tolist(), index=1)
        fig_fs = px.scatter(df.reset_index(), x=feat_x, y=feat_y,
            opacity=0.4, color_discrete_sequence=['#58a6ff'],
            title=f'{feat_x} vs {feat_y}', trendline='ols')
        fig_fs.update_layout(**PLOTLY_LAYOUT, height=400)
        st.plotly_chart(fig_fs, width="stretch")

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
        st.plotly_chart(fig_decomp, width="stretch")

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
        st.plotly_chart(fig_acf, width="stretch")

with nav_tab3:
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
    st.plotly_chart(fig_avp, width="stretch")

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
    st.plotly_chart(fig_split, width="stretch")

    col_info1, col_info2 = st.columns(2)
    col_info1.markdown(
        f"<div class='insight-box'><strong>Training samples:</strong> {len(X_train):,} rows (80%)</div>",
        unsafe_allow_html=True
    )
    col_info2.markdown(
        f"<div class='insight-box'><strong>Test samples:</strong> {len(X_test):,} rows (20%)</div>",
        unsafe_allow_html=True
    )
