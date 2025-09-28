"""
Data fetching module for the Investment Portfolio Recommendation Engine.
Supports Alpha Vantage and yfinance (Yahoo Finance) free data sources.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import streamlit as st
from loguru import logger

from .config import config


class RateLimiter:
    """Simple rate limiter for API calls to respect free tier limits."""
    
    def __init__(self):
        self.call_times = {}
    
    def can_make_call(self, api_name: str, calls_per_minute: int = None, calls_per_day: int = None) -> bool:
        """Check if we can make an API call based on rate limits."""
        now = time.time()
        
        if api_name not in self.call_times:
            self.call_times[api_name] = []
        
        # Clean old calls
        if calls_per_minute:
            minute_ago = now - 60
            self.call_times[api_name] = [t for t in self.call_times[api_name] if t > minute_ago]
            
        if calls_per_day:
            day_ago = now - 86400
            self.call_times[api_name] = [t for t in self.call_times[api_name] if t > day_ago]
        
        # Check limits
        current_calls = len(self.call_times[api_name])
        if calls_per_minute and current_calls >= calls_per_minute:
            return False
        if calls_per_day and current_calls >= calls_per_day:
            return False
        
        return True
    
    def record_call(self, api_name: str):
        """Record that an API call was made."""
        if api_name not in self.call_times:
            self.call_times[api_name] = []
        self.call_times[api_name].append(time.time())


class DataFetcher:
    """Main data fetching class that handles Alpha Vantage and yfinance data sources."""
    
    def __init__(self):
        self.rate_limiter = RateLimiter()
    
    def get_stock_data_yfinance(self, tickers: List[str], period: str = "1y") -> pd.DataFrame:
        """Fetch stock data using yfinance (completely free)."""
        try:
            logger.info(f"Fetching data for {tickers} using yfinance")
            
            # Download data for all tickers
            data = yf.download(tickers, period=period, group_by='ticker', auto_adjust=True, prepost=True, threads=True)
            
            if len(tickers) == 1:
                # Single ticker case
                prices = data['Close'].to_frame()
                prices.columns = tickers
            else:
                # Multiple tickers case
                prices = pd.DataFrame()
                for ticker in tickers:
                    if ticker in data.columns.levels[0]:
                        prices[ticker] = data[ticker]['Close']
            
            # Forward fill missing values and drop any remaining NaN
            prices = prices.fillna(method='ffill').dropna()
            
            logger.info(f"Successfully fetched {len(prices)} data points for {len(prices.columns)} tickers")
            return prices
            
        except Exception as e:
            logger.error(f"Error fetching data with yfinance: {e}")
            return pd.DataFrame()
    
    def get_current_price_yfinance(self, ticker: str) -> Optional[float]:
        """Get current price for a single ticker using yfinance."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return info.get('currentPrice') or info.get('regularMarketPrice')
        except Exception as e:
            logger.error(f"Error getting current price for {ticker}: {e}")
            return None
    
    def get_stock_info_yfinance(self, ticker: str) -> Dict:
        """Get detailed stock information using yfinance."""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            return {
                'symbol': ticker,
                'name': info.get('longName', ticker),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE'),
                'dividend_yield': info.get('dividendYield'),
                'current_price': info.get('currentPrice') or info.get('regularMarketPrice'),
                'previous_close': info.get('previousClose'),
                'day_change': info.get('regularMarketChange'),
                'day_change_percent': info.get('regularMarketChangePercent')
            }
        except Exception as e:
            logger.error(f"Error getting info for {ticker}: {e}")
            return {'symbol': ticker, 'name': ticker}
    
    def get_portfolio_data(self, tickers: List[str], period: str = "1y") -> pd.DataFrame:
        """Get portfolio data using yfinance as primary source."""
        # Use yfinance (free and reliable)
        data = self.get_stock_data_yfinance(tickers, period)
        return data
    
    def get_current_prices(self, tickers: List[str]) -> Dict[str, float]:
        """Get current prices for a list of tickers using yfinance."""
        prices = {}
        
        for ticker in tickers:
            price = self.get_current_price_yfinance(ticker)
            if price is not None:
                prices[ticker] = price
            else:
                logger.warning(f"Could not fetch current price for {ticker}")
        
        return prices
    
    def get_market_data_summary(self, tickers: List[str]) -> pd.DataFrame:
        """Get a summary of market data for the given tickers."""
        summary_data = []
        
        for ticker in tickers:
            info = self.get_stock_info_yfinance(ticker)
            summary_data.append(info)
        
        return pd.DataFrame(summary_data)
    
    def get_risk_free_rate(self) -> float:
        """Get the current risk-free rate (10-year Treasury rate)."""
        try:
            # Fetch 10-year Treasury rate using yfinance
            treasury = yf.Ticker("^TNX")
            hist = treasury.history(period="5d")
            if not hist.empty:
                return hist['Close'].iloc[-1] / 100  # Convert percentage to decimal
        except Exception as e:
            logger.warning(f"Could not fetch risk-free rate: {e}")
        
        # Return configured default rate
        return config.RISK_FREE_RATE
    
    def validate_tickers(self, tickers: List[str]) -> List[str]:
        """Validate that tickers exist and return only valid ones."""
        valid_tickers = []
        
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                # Check if ticker has basic information
                if info and ('currentPrice' in info or 'regularMarketPrice' in info):
                    valid_tickers.append(ticker)
                else:
                    logger.warning(f"Ticker {ticker} may not be valid or has no price data")
            except Exception as e:
                logger.warning(f"Error validating ticker {ticker}: {e}")
        
        return valid_tickers


# Create global data fetcher instance
data_fetcher = DataFetcher()