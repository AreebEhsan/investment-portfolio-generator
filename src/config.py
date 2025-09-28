"""
Configuration module for the Investment Portfolio Recommendation Engine.
Handles environment variables and application settings.
"""

import os
from dotenv import load_dotenv
from typing import Optional
import streamlit as st

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for managing app settings and API keys."""
    
    # API Keys (Free tiers only)
    ALPHA_VANTAGE_API_KEY: Optional[str] = os.getenv('ALPHA_VANTAGE_API_KEY')
    NEWS_API_KEY: Optional[str] = os.getenv('NEWS_API_KEY')
    
    # Application Settings
    APP_DEBUG: bool = os.getenv('APP_DEBUG', 'False').lower() == 'true'
    CACHE_TTL_SECONDS: int = int(os.getenv('CACHE_TTL_SECONDS', '600'))
    UPDATE_INTERVAL_SECONDS: int = int(os.getenv('UPDATE_INTERVAL_SECONDS', '60'))
    DEFAULT_PORTFOLIO_SIZE: float = float(os.getenv('DEFAULT_PORTFOLIO_SIZE', '10000'))
    
    # Financial Settings
    RISK_FREE_RATE: float = float(os.getenv('RISK_FREE_RATE', '0.04'))
    
    # Data Provider Settings
    PRIMARY_DATA_PROVIDER: str = os.getenv('PRIMARY_DATA_PROVIDER', 'yfinance')
    SECONDARY_DATA_PROVIDER: str = os.getenv('SECONDARY_DATA_PROVIDER', 'alpha_vantage')
    
    # Rate Limiting (Free tier limits)
    ALPHA_VANTAGE_CALLS_PER_DAY: int = int(os.getenv('ALPHA_VANTAGE_CALLS_PER_DAY', '20'))
    NEWS_API_CALLS_PER_DAY: int = int(os.getenv('NEWS_API_CALLS_PER_DAY', '90'))
    
    # Logging
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    LOG_TO_FILE: bool = os.getenv('LOG_TO_FILE', 'True').lower() == 'true'
    
    # Default asset lists (popular free assets)
    DEFAULT_STOCKS = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 
        'NVDA', 'META', 'BRK-B', 'V', 'JNJ'
    ]
    
    DEFAULT_ETFS = [
        'SPY', 'QQQ', 'IWM', 'VTI', 'VXUS', 
        'BND', 'VNQ', 'GLD', 'XLE', 'XLF'
    ]
    
    DEFAULT_CRYPTO = [
        'BTC-USD', 'ETH-USD', 'ADA-USD', 'DOT-USD', 'LINK-USD'
    ]
    
    # Sector mappings for diversification
    SECTOR_TICKERS = {
        'Technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META'],
        'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'TMO'],
        'Finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS'],
        'Consumer': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE'],
        'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB'],
        'Real Estate': ['VNQ', 'PLD', 'AMT', 'CCI', 'EQIX']
    }
    
    @classmethod
    def validate_api_keys(cls) -> dict:
        """Validate which API keys are available."""
        validation = {
            'yfinance': True,  # Always available (no key needed)
            'alpha_vantage': cls.ALPHA_VANTAGE_API_KEY is not None and cls.ALPHA_VANTAGE_API_KEY != 'your_free_alpha_vantage_api_key_here',
            'news_api': cls.NEWS_API_KEY is not None and cls.NEWS_API_KEY != 'your_free_news_api_key_here'
        }
        return validation
    
    @classmethod
    def get_available_providers(cls) -> list:
        """Get list of available data providers based on API key validation."""
        validation = cls.validate_api_keys()
        providers = []
        
        if validation['yfinance']:
            providers.append('yfinance')
        if validation['alpha_vantage']:
            providers.append('alpha_vantage')
            
        return providers
    
    @classmethod
    def show_api_status(cls):
        """Display API key status in Streamlit sidebar."""
        validation = cls.validate_api_keys()
        
        st.sidebar.subheader("📊 Data Sources Status")
        
        # YFinance (always available)
        st.sidebar.success("✅ Yahoo Finance (yfinance) - Free")
        
        # Alpha Vantage
        if validation['alpha_vantage']:
            st.sidebar.success("✅ Alpha Vantage - Free Tier")
        else:
            st.sidebar.warning("⚠️ Alpha Vantage - Not configured")
        
        # News API
        if validation['news_api']:
            st.sidebar.success("✅ News API - Free Tier")
        else:
            st.sidebar.info("ℹ️ News API - Not configured (optional)")

# Create global config instance
config = Config()