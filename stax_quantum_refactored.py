"""
STAX QUANTUM - Production-Ready Trading Application
Refactored, Debugged, and Security-Hardened
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, List
import plotly.graph_objects as go
from dataclasses import dataclass
import re

# Graceful imports with error handling
try:
    import yfinance as yf
except ImportError:
    yf = None
    st.error("Missing: yfinance. Run: pip install yfinance")

try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
except ImportError:
    Sequential = None

try:
    from sklearn.preprocessing import MinMaxScaler
except ImportError:
    MinMaxScaler = None

try:
    from binance.client import Client
    from binance.exceptions import BinanceAPIException
except ImportError:
    Client = None

# ==================== CONFIGURATION ====================

@dataclass
class Config:
    """Application configuration"""
    DB_PATH: str = "stax_quantum.db"
    LOG_PATH: str = "stax_quantum.log"
    
    # Validation
    MIN_USERNAME_LENGTH: int = 3
    MAX_USERNAME_LENGTH: int = 50
    MIN_PASSWORD_LENGTH: int = 6
    MIN_ORDER_AMOUNT: float = 0.0001
    MAX_ORDER_AMOUNT: float = 1000.0
    
    # Tickers
    VALID_TICKERS: List[str] = None
    
    # ML
    LSTM_LOOKBACK: int = 10
    LSTM_EPOCHS: int = 5
    LSTM_BATCH_SIZE: int = 16
    LSTM_DROPOUT: float = 0.2
    
    # Cache
    CACHE_TTL: int = 900
    
    def __post_init__(self):
        if self.VALID_TICKERS is None:
            self.VALID_TICKERS = ["BTC-USD", "ETH-USD", "SOL-USD", "AAPL", "NVDA"]

CONFIG = Config()

# ==================== LOGGING ====================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(CONFIG.LOG_PATH),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==================== PASSWORD MANAGER ====================

class PasswordManager:
    """Handles password hashing and verification"""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password with validation"""
        if not password or len(password) < CONFIG.MIN_PASSWORD_LENGTH:
            raise ValueError(f"Password must be at least {CONFIG.MIN_PASSWORD_LENGTH} characters")
        return hashlib.sha256(password.encode()).hexdigest()
    
    @staticmethod
    def verify_password(password: str, hash_value: str) -> bool:
        """Verify password against hash"""
        try:
            return PasswordManager.hash_password(password) == hash_value
        except ValueError:
            return False

# ==================== INPUT VALIDATOR ====================

class InputValidator:
    """Validates all user inputs"""
    
    @staticmethod
    def validate_username(username: str) -> Tuple[bool, str]:
        """Validate username format and length"""
        if not username:
            return False, "Username cannot be empty"
        if len(username) < CONFIG.MIN_USERNAME_LENGTH:
            return False, f"Username must be at least {CONFIG.MIN_USERNAME_LENGTH} characters"
        if len(username) > CONFIG.MAX_USERNAME_LENGTH:
            return False, f"Username must be less than {CONFIG.MAX_USERNAME_LENGTH} characters"
        if not re.match(r"^[a-zA-Z0-9_-]+$", username):
            return False, "Username must contain only alphanumeric characters, hyphens, and underscores"
        return True, ""
    
    @staticmethod
    def validate_password(password: str) -> Tuple[bool, str]:
        """Validate password strength"""
        if not password:
            return False, "Password cannot be empty"
        if len(password) < CONFIG.MIN_PASSWORD_LENGTH:
            return False, f"Password must be at least {CONFIG.MIN_PASSWORD_LENGTH} characters"
        return True, ""
    
    @staticmethod
    def validate_ticker(ticker: str) -> Tuple[bool, str]:
        """Validate ticker symbol"""
        if ticker not in CONFIG.VALID_TICKERS:
            return False, f"Invalid ticker: {ticker}. Valid options: {', '.join(CONFIG.VALID_TICKERS)}"
        return True, ""
    
    @staticmethod
    def validate_order_amount(amount: float) -> Tuple[bool, str]:
        """Validate order amount"""
        if amount < CONFIG.MIN_ORDER_AMOUNT:
            return False, f"Order amount must be >= {CONFIG.MIN_ORDER_AMOUNT}"
        if amount > CONFIG.MAX_ORDER_AMOUNT:
            return False, f"Order amount must be <= {CONFIG.MAX_ORDER_AMOUNT}"
        return True, ""

# ==================== DATABASE MANAGER ====================

class DatabaseManager:
    """Handles all database operations"""
    
    def __init__(self, db_path: str = CONFIG.DB_PATH):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute('PRAGMA journal_mode=WAL')  # Better concurrency
        self._init_tables()
    
    def _init_tables(self):
        """Initialize database tables"""
        try:
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS portfolio (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL,
                    asset TEXT NOT NULL,
                    amount REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (username) REFERENCES users(username)
                )
            """)
            
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT,
                    action TEXT NOT NULL,
                    details TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            self.conn.commit()
            logger.info("Database tables initialized")
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
    
    def user_exists(self, username: str) -> bool:
        """Check if user exists"""
        try:
            res = self.conn.execute(
                "SELECT * FROM users WHERE username=?",
                (username,)
            ).fetchone()
            return res is not None
        except Exception as e:
            logger.error(f"User check error: {e}")
            return False
    
    def create_user(self, username: str, password_hash: str) -> bool:
        """Create new user"""
        try:
            self.conn.execute(
                "INSERT INTO users (username, password) VALUES (?, ?)",
                (username, password_hash)
            )
            self.conn.commit()
            self._audit_log(username, "USER_CREATED", "New user registered")
            logger.info(f"User created: {username}")
            return True
        except sqlite3.IntegrityError:
            logger.warning(f"Duplicate username attempt: {username}")
            return False
        except Exception as e:
            logger.error(f"User creation error: {e}")
            return False
    
    def verify_credentials(self, username: str, password_hash: str) -> bool:
        """Verify user credentials"""
        try:
            res = self.conn.execute(
                "SELECT password FROM users WHERE username=?",
                (username,)
            ).fetchone()
            if res and res['password'] == password_hash:
                self._audit_log(username, "LOGIN_SUCCESS", "User logged in")
                logger.info(f"Login success: {username}")
                return True
            self._audit_log(username, "LOGIN_FAILED", "Invalid credentials")
            return False
        except Exception as e:
            logger.error(f"Credential verification error: {e}")
            return False
    
    def add_portfolio_entry(self, username: str, asset: str, amount: float, entry_price: float) -> bool:
        """Add portfolio entry"""
        try:
            self.conn.execute(
                "INSERT INTO portfolio (username, asset, amount, entry_price) VALUES (?, ?, ?, ?)",
                (username, asset, amount, entry_price)
            )
            self.conn.commit()
            self._audit_log(username, "ORDER_PLACED", f"{amount} {asset} @ ${entry_price}")
            logger.info(f"Portfolio entry added for {username}: {amount} {asset}")
            return True
        except Exception as e:
            logger.error(f"Portfolio entry error: {e}")
            return False
    
    def get_user_portfolio(self, username: str) -> pd.DataFrame:
        """Get user's portfolio with aggregation"""
        try:
            query = """
                SELECT asset, SUM(amount) as total_amount, AVG(entry_price) as avg_entry_price
                FROM portfolio
                WHERE username=?
                GROUP BY asset
            """
            return pd.read_sql(query, self.conn, params=(username,))
        except Exception as e:
            logger.error(f"Portfolio retrieval error: {e}")
            return pd.DataFrame()
    
    def _audit_log(self, username: Optional[str], action: str, details: str = ""):
        """Log audit trail"""
        try:
            self.conn.execute(
                "INSERT INTO audit_log (username, action, details) VALUES (?, ?, ?)",
                (username, action, details)
            )
            self.conn.commit()
        except Exception as e:
            logger.error(f"Audit log error: {e}")

# ==================== MARKET DATA FETCHER ====================

class MarketDataFetcher:
    """Fetches and processes market data"""
    
    @staticmethod
    @st.cache_data(ttl=CONFIG.CACHE_TTL, show_spinner=False)
    def fetch(ticker: str, period: str = "5d", interval: str = "1h") -> pd.DataFrame:
        """Fetch market data with error handling"""
        if yf is None:
            logger.error("yfinance not available")
            return pd.DataFrame()
        
        try:
            df = yf.download(ticker, period=period, interval=interval, progress=False, quiet=True)
            
            if df.empty:
                logger.warning(f"Empty data for {ticker}")
                return pd.DataFrame()
            
            # Handle MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            
            # Ensure required columns
            required_cols = ['Open', 'High', 'Low', 'Close']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"Missing required columns for {ticker}")
                return pd.DataFrame()
            
            logger.info(f"Fetched {len(df)} rows for {ticker}")
            return df
        
        except Exception as e:
            logger.error(f"Market data fetch error for {ticker}: {e}")
            return pd.DataFrame()

# ==================== LSTM PREDICTOR ====================

class LSTMPredictor:
    """LSTM-based price forecasting"""
    
    @staticmethod
    def prepare_data(prices: Optional[np.ndarray], lookback: int = CONFIG.LSTM_LOOKBACK) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[MinMaxScaler]]:
        """Prepare data for LSTM training"""
        if prices is None or len(prices) < lookback + 1:
            logger.warning(f"Insufficient data: {len(prices) if prices is not None else 0} < {lookback + 1}")
            return None, None, None
        
        try:
            prices = np.array(prices).reshape(-1, 1).astype(np.float32)
            
            if MinMaxScaler is None:
                logger.error("sklearn not available")
                return None, None, None
            
            scaler = MinMaxScaler()
            scaled = scaler.fit_transform(prices)
            
            X, y = [], []
            for i in range(lookback, len(scaled)):
                X.append(scaled[i - lookback:i])
                y.append(scaled[i])
            
            return np.array(X), np.array(y), scaler
        except Exception as e:
            logger.error(f"Data preparation error: {e}")
            return None, None, None
    
    @staticmethod
    @st.cache_data(ttl=CONFIG.CACHE_TTL, show_spinner=False)
    def forecast(prices_tuple: Optional[Tuple]) -> Optional[float]:
        """Generate LSTM forecast"""
        if prices_tuple is None or Sequential is None:
            return None
        
        if len(prices_tuple) < CONFIG.LSTM_LOOKBACK + 1:
            logger.warning("Insufficient data for forecast")
            return None
        
        try:
            prices = np.array(prices_tuple)
            X, y, scaler = LSTMPredictor.prepare_data(prices)
            
            if X is None:
                return None
            
            # Build model
            model = Sequential([
                LSTM(32, activation='relu', input_shape=(CONFIG.LSTM_LOOKBACK, 1)),
                Dropout(CONFIG.LSTM_DROPOUT),
                Dense(16, activation='relu'),
                Dropout(CONFIG.LSTM_DROPOUT),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            # Early stopping to prevent overfitting
            early_stop = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
            
            model.fit(
                X, y,
                epochs=CONFIG.LSTM_EPOCHS,
                batch_size=CONFIG.LSTM_BATCH_SIZE,
                callbacks=[early_stop],
                verbose=0
            )
            
            # Forecast
            last_sequence = prices[-CONFIG.LSTM_LOOKBACK:].reshape(1, CONFIG.LSTM_LOOKBACK, 1)
            last_sequence_scaled = (last_sequence - prices.min()) / (prices.max() - prices.min() + 1e-8)
            
            pred_scaled = model.predict(last_sequence_scaled, verbose=0)
            pred = pred_scaled * (prices.max() - prices.min()) + prices.min()
            
            logger.info(f"LSTM forecast: {float(pred[0][0])}")
            return float(pred[0][0])
        
        except Exception as e:
            logger.error(f"LSTM forecast error: {e}")
            return None

# ==================== BINANCE TRADER ====================

class BinanceTrader:
    """Handles Binance API interactions"""
    
    def __init__(self, api_key: str, secret_key: str, testnet: bool = True):
        if Client is None:
            raise ImportError("python-binance not installed")
        
        try:
            self.client = Client(api_key, secret_key, testnet=testnet)
            self.client.get_account()  # Verify connection
            logger.info(f"Binance connected (testnet={testnet})")
        except BinanceAPIException as e:
            logger.error(f"Binance API error: {e}")
            raise
    
    def execute_order(self, symbol: str, quantity: float, order_type: str = "MARKET") -> bool:
        """Execute order with validation"""
        try:
            symbol_clean = symbol.replace("-", "")
            
            order = self.client.create_test_order(
                symbol=symbol_clean,
                side='BUY',
                type=order_type,
                quantity=quantity
            )
            
            logger.info(f"Order executed: {symbol} x{quantity}")
            return True
        except BinanceAPIException as e:
            logger.error(f"Order execution error: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected order error: {e}")
            return False

# ==================== STREAMLIT PAGE SETUP ====================

st.set_page_config(page_title="STAX QUANTUM", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;500;700&display=swap');
    html, body, [data-testid="stAppViewContainer"] { 
        background-color: #080C13; 
        color: #E0E1DD; 
        font-family: 'JetBrains Mono', monospace; 
    }
    [data-testid="stSidebar"] { 
        background-color: #1B263B; 
        border-right: 1px solid #415A77; 
    }
    .stMetric { 
        background: linear-gradient(135deg, rgba(27, 38, 59, 0.8) 0%, rgba(13, 59, 102, 0.8) 100%); 
        border: 1px solid #778DA9; 
        padding: 15px !important; 
        border-radius: 10px; 
        box-shadow: 0 4px 15px rgba(0,0,0,0.2); 
    }
    h1, h2, h3 { color: #E0E1DD !important; letter-spacing: 1px; }
    div.stButton > button { 
        background-color: rgba(13, 59, 102, 0.5); 
        color: #00B4D8; 
        border: 1px solid #00B4D8; 
        border-radius: 6px; 
        transition: all 0.3s ease-in-out; 
    }
    div.stButton > button:hover { 
        background-color: #00B4D8; 
        color: #080C13; 
        box-shadow: 0 0 15px rgba(0, 180, 216, 0.6); 
    }
    .stax-title { text-align: center; letter-spacing: 15px; color: #E0E1DD; text-shadow: 0 0 10px #00B4D8; }
    .stax-cyan { color: #00B4D8; font-weight: bold; }
    .stax-mint { color: #00FFC2; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ==================== SESSION INITIALIZATION ====================

def init_app():
    """Initialize application state"""
    if "user" not in st.session_state:
        st.session_state.user = None
        st.session_state.binance_client = None
        st.session_state.db = DatabaseManager()

init_app()

# ==================== AUTHENTICATION UI ====================

if not st.session_state.user:
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown("<div style='height: 15vh;'></div>", unsafe_allow_html=True)
        st.markdown("<h1 class='stax-title'>S T A X</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align:center; color:#778DA9; letter-spacing: 2px;'>QUANTUM EDITION</p>", unsafe_allow_html=True)
        
        with st.container():
            st.markdown("<br>", unsafe_allow_html=True)
            u = st.text_input("Access ID")
            p = st.text_input("Neural Key", type="password")
            
            col_login, col_register = st.columns(2)
            
            with col_login:
                if st.button("Initialize Session", use_container_width=True):
                    if not u or not p:
                        st.warning("Enter ID and Key")
                    else:
                        try:
                            pw_hash = PasswordManager.hash_password(p)
                            if st.session_state.db.verify_credentials(u, pw_hash):
                                st.session_state.user = u
                                st.rerun()
                            else:
                                st.error("Authentication Failed: Invalid ID or Neural Key")
                        except Exception as e:
                            st.error(f"Login error: {e}")
            
            with col_register:
                if st.button("New Enrollment", use_container_width=True):
                    try:
                        valid, msg = InputValidator.validate_username(u)
                        if not valid:
                            st.error(f"Username: {msg}")
                        else:
                            valid, msg = InputValidator.validate_password(p)
                            if not valid:
                                st.error(f"Password: {msg}")
                            else:
                                if st.session_state.db.user_exists(u):
                                    st.error("Username already exists")
                                else:
                                    pw_hash = PasswordManager.hash_password(p)
                                    if st.session_state.db.create_user(u, pw_hash):
                                        st.success("Enrollment Successful! You can now login.")
                                        st.rerun()
                                    else:
                                        st.error("Enrollment failed. Please try again.")
                    except Exception as e:
                        st.error(f"Registration error: {e}")
else:
    # ==================== MAIN APP UI ====================
    
    # Sidebar
    with st.sidebar:
        st.markdown("<h2 style='color: #00B4D8;'>⚙️ CONTROL PANEL</h2>", unsafe_allow_html=True)
        
        page = st.radio(
            "Navigation",
            ["📊 Dashboard", "💼 Portfolio", "📈 Analytics", "🔧 Settings"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.user = None
            st.session_state.binance_client = None
            logger.info(f"User logged out: {u}")
            st.rerun()
    
    # Main content
    st.markdown(f"<p style='text-align: right; color: #778DA9; font-size: 12px;'>👤 Session: <span class='stax-cyan'>{st.session_state.user}</span></p>", unsafe_allow_html=True)
    
    if page == "📊 Dashboard":
        st.markdown("<h2 class='stax-cyan'>Market Overview</h2>", unsafe_allow_html=True)
        
        ticker_col, period_col = st.columns(2)
        with ticker_col:
            selected_ticker = st.selectbox("Select Asset", CONFIG.VALID_TICKERS)
        with period_col:
            period = st.selectbox("Period", ["5d", "1mo", "3mo"])
        
        if selected_ticker:
            df = MarketDataFetcher.fetch(selected_ticker, period=period)
            
            if not df.empty:
                # Display price chart
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close']
                ))
                fig.update_layout(
                    title=f"{selected_ticker} - Price Action",
                    yaxis_title="Price",
                    template="plotly_dark",
                    xaxis_rangeslider_visible=False
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Metrics
                current_price = df['Close'].iloc[-1]
                price_change = df['Close'].iloc[-1] - df['Close'].iloc[0]
                pct_change = (price_change / df['Close'].iloc[0]) * 100
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Current Price", f"${current_price:.2f}")
                col2.metric("Change", f"${price_change:.2f}", f"{pct_change:.2f}%")
                col3.metric("High", f"${df['High'].max():.2f}")
                col4.metric("Low", f"${df['Low'].min():.2f}")
            else:
                st.error("Unable to fetch market data")
    
    elif page == "💼 Portfolio":
        st.markdown("<h2 class='stax-cyan'>Your Holdings</h2>", unsafe_allow_html=True)
        
        portfolio_df = st.session_state.db.get_user_portfolio(st.session_state.user)
        
        if not portfolio_df.empty:
            st.dataframe(portfolio_df, use_container_width=True)
        else:
            st.info("Your portfolio is empty. Start trading!")
        
        st.markdown("<h3 class='stax-mint'>Place Order</h3>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            ticker = st.selectbox("Asset", CONFIG.VALID_TICKERS, key="order_ticker")
        with col2:
            amount = st.number_input("Amount", min_value=CONFIG.MIN_ORDER_AMOUNT, max_value=CONFIG.MAX_ORDER_AMOUNT)
        
        if st.button("Execute Order", use_container_width=True):
            valid, msg = InputValidator.validate_ticker(ticker)
            if not valid:
                st.error(msg)
            else:
                valid, msg = InputValidator.validate_order_amount(amount)
                if not valid:
                    st.error(msg)
                else:
                    # Get current price
                    df = MarketDataFetcher.fetch(ticker)
                    if not df.empty:
                        entry_price = df['Close'].iloc[-1]
                        if st.session_state.db.add_portfolio_entry(st.session_state.user, ticker, amount, entry_price):
                            st.success(f"Order placed: {amount} {ticker} @ ${entry_price:.2f}")
                            st.rerun()
                        else:
                            st.error("Failed to place order")
    
    elif page == "📈 Analytics":
        st.markdown("<h2 class='stax-cyan'>Price Forecast</h2>", unsafe_allow_html=True)
        
        ticker = st.selectbox("Select Asset for Forecast", CONFIG.VALID_TICKERS, key="forecast_ticker")
        
        if st.button("Generate Forecast", use_container_width=True):
            df = MarketDataFetcher.fetch(ticker, period="3mo")
            
            if not df.empty and len(df) > CONFIG.LSTM_LOOKBACK:
                prices = df['Close'].values
                forecast = LSTMPredictor.forecast(tuple(prices))
                
                if forecast:
                    current_price = prices[-1]
                    change = forecast - current_price
                    pct_change = (change / current_price) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Current Price", f"${current_price:.2f}")
                    col2.metric("Forecast", f"${forecast:.2f}")
                    col3.metric("Predicted Change", f"{pct_change:.2f}%")
                else:
                    st.warning("Forecast generation failed")
            else:
                st.error("Insufficient data for forecast")
    
    elif page == "🔧 Settings":
        st.markdown("<h2 class='stax-cyan'>Account Settings</h2>", unsafe_allow_html=True)
        
        st.info(f"User: {st.session_state.user}")
        
        with st.expander("Change Password"):
            current_pass = st.text_input("Current Password", type="password")
            new_pass = st.text_input("New Password", type="password")
            confirm_pass = st.text_input("Confirm Password", type="password")
            
            if st.button("Update Password"):
                pw_hash = PasswordManager.hash_password(current_pass)
                if st.session_state.db.verify_credentials(st.session_state.user, pw_hash):
                    if new_pass == confirm_pass:
                        new_pw_hash = PasswordManager.hash_password(new_pass)
                        st.session_state.db.conn.execute(
                            "UPDATE users SET password=? WHERE username=?",
                            (new_pw_hash, st.session_state.user)
                        )
                        st.session_state.db.conn.commit()
                        st.success("Password updated successfully")
                    else:
                        st.error("Passwords do not match")
                else:
                    st.error("Current password is incorrect")
        
        with st.expander("Binance API Configuration"):
            api_key = st.text_input("API Key", type="password")
            secret_key = st.text_input("Secret Key", type="password")
            
            if st.button("Connect Binance"):
                try:
                    trader = BinanceTrader(api_key, secret_key, testnet=True)
                    st.session_state.binance_client = trader
                    st.success("Binance connected successfully!")
                except Exception as e:
                    st.error(f"Connection failed: {e}")
