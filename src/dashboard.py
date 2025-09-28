"""
Dashboard module with reusable chart components for the Investment Portfolio Recommendation Engine.
Contains Plotly-based visualizations for portfolio analysis and market data.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger

from .config import config


class ChartGenerator:
    """Generate various charts for portfolio analysis and market visualization."""
    
    def __init__(self):
        """Initialize chart generator with default styling."""
        self.color_palette = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        
        self.chart_config = {
            'displayModeBar': False,
            'staticPlot': False,
            'responsive': True
        }
    
    def create_portfolio_allocation_pie(self, weights: np.ndarray, tickers: List[str], 
                                       title: str = "Portfolio Allocation") -> go.Figure:
        """
        Create a pie chart showing portfolio allocation.
        
        Args:
            weights: Portfolio weights array
            tickers: List of ticker symbols
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        # Filter out very small weights for cleaner visualization
        min_weight = 0.01  # 1%
        
        filtered_data = []
        other_weight = 0
        
        for ticker, weight in zip(tickers, weights):
            if weight >= min_weight:
                filtered_data.append((ticker, weight))
            else:
                other_weight += weight
        
        if other_weight > 0:
            filtered_data.append(("Others", other_weight))
        
        labels, values = zip(*filtered_data) if filtered_data else ([], [])
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            textinfo='label+percent',
            textfont_size=12,
            marker=dict(
                colors=self.color_palette[:len(labels)],
                line=dict(color='#FFFFFF', width=2)
            )
        )])
        
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.01
            ),
            margin=dict(t=50, b=20, l=20, r=120),
            height=400
        )
        
        return fig
    
    def create_price_chart(self, price_data: pd.DataFrame, title: str = "Price History",
                          show_volume: bool = False) -> go.Figure:
        """
        Create a line chart showing price history for multiple assets.
        
        Args:
            price_data: DataFrame with price data
            title: Chart title
            show_volume: Whether to show volume data (if available)
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        for i, column in enumerate(price_data.columns):
            fig.add_trace(go.Scatter(
                x=price_data.index,
                y=price_data[column],
                mode='lines',
                name=column,
                line=dict(
                    color=self.color_palette[i % len(self.color_palette)],
                    width=2
                ),
                hovertemplate=f'{column}: $%{{y:.2f}}<br>Date: %{{x}}<extra></extra>'
            ))
        
        fig.update_layout(
            title={
                'text': title,
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18}
            },
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.2)",
                borderwidth=1
            ),
            margin=dict(t=50, b=40, l=60, r=40),
            height=500
        )
        
        # Add range selector
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=False),
                type="date"
            )
        )
        
        return fig
    
    def create_candlestick_chart(self, ohlc_data: pd.DataFrame, ticker: str) -> go.Figure:
        """
        Create a candlestick chart for a single asset.
        
        Args:
            ohlc_data: DataFrame with OHLC data
            ticker: Ticker symbol
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure(data=[go.Candlestick(
            x=ohlc_data.index,
            open=ohlc_data['Open'],
            high=ohlc_data['High'],
            low=ohlc_data['Low'],
            close=ohlc_data['Close'],
            name=ticker
        )])
        
        fig.update_layout(
            title=f'{ticker} Candlestick Chart',
            xaxis_title="Date",
            yaxis_title="Price ($)",
            xaxis_rangeslider_visible=False,
            height=500
        )
        
        return fig
    
    def create_returns_histogram(self, returns: pd.Series, ticker: str) -> go.Figure:
        """
        Create a histogram of returns distribution.
        
        Args:
            returns: Series of returns
            ticker: Ticker symbol
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure(data=[go.Histogram(
            x=returns * 100,  # Convert to percentage
            nbinsx=50,
            name=f'{ticker} Returns',
            marker_color='skyblue',
            opacity=0.7
        )])
        
        # Add normal distribution overlay
        mean_return = returns.mean() * 100
        std_return = returns.std() * 100
        x_norm = np.linspace(returns.min() * 100, returns.max() * 100, 100)
        y_norm = (1 / (std_return * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_norm - mean_return) / std_return) ** 2)
        y_norm = y_norm * len(returns) * (returns.max() * 100 - returns.min() * 100) / 50  # Scale to histogram
        
        fig.add_trace(go.Scatter(
            x=x_norm,
            y=y_norm,
            mode='lines',
            name='Normal Distribution',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title=f'{ticker} Returns Distribution',
            xaxis_title="Daily Returns (%)",
            yaxis_title="Frequency",
            height=400,
            showlegend=True
        )
        
        return fig
    
    def create_correlation_heatmap(self, correlation_matrix: pd.DataFrame) -> go.Figure:
        """
        Create a correlation heatmap.
        
        Args:
            correlation_matrix: Correlation matrix DataFrame
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=np.round(correlation_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Asset Correlation Matrix',
            height=500,
            margin=dict(t=50, b=50, l=100, r=50)
        )
        
        return fig
    
    def create_efficient_frontier(self, frontier_data: pd.DataFrame, 
                                 optimal_portfolio: Optional[Dict] = None) -> go.Figure:
        """
        Create efficient frontier chart.
        
        Args:
            frontier_data: DataFrame with efficient frontier data
            optimal_portfolio: Optional optimal portfolio point to highlight
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Plot efficient frontier
        fig.add_trace(go.Scatter(
            x=frontier_data['volatility'] * 100,
            y=frontier_data['return'] * 100,
            mode='lines+markers',
            name='Efficient Frontier',
            line=dict(color='blue', width=3),
            marker=dict(size=4)
        ))
        
        # Highlight optimal portfolio if provided
        if optimal_portfolio and optimal_portfolio.get('optimization_success'):
            fig.add_trace(go.Scatter(
                x=[optimal_portfolio['volatility'] * 100],
                y=[optimal_portfolio['expected_return'] * 100],
                mode='markers',
                name='Optimal Portfolio',
                marker=dict(
                    size=15,
                    color='red',
                    symbol='star',
                    line=dict(color='darkred', width=2)
                ),
                hovertemplate=f"Return: {optimal_portfolio['expected_return']*100:.2f}%<br>" +
                             f"Volatility: {optimal_portfolio['volatility']*100:.2f}%<br>" +
                             f"Sharpe Ratio: {optimal_portfolio['sharpe_ratio']:.3f}<extra></extra>"
            ))
        
        fig.update_layout(
            title='Efficient Frontier',
            xaxis_title='Volatility (%)',
            yaxis_title='Expected Return (%)',
            height=500,
            showlegend=True,
            hovermode='closest'
        )
        
        return fig
    
    def create_portfolio_value_chart(self, portfolio_values: pd.Series, 
                                   benchmark_values: Optional[pd.Series] = None) -> go.Figure:
        """
        Create portfolio value over time chart with optional benchmark comparison.
        
        Args:
            portfolio_values: Series of portfolio values over time
            benchmark_values: Optional benchmark values for comparison
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Portfolio line
        fig.add_trace(go.Scatter(
            x=portfolio_values.index,
            y=portfolio_values.values,
            mode='lines',
            name='Portfolio',
            line=dict(color='blue', width=3),
            hovertemplate='Portfolio: $%{y:,.2f}<br>Date: %{x}<extra></extra>'
        ))
        
        # Benchmark line if provided
        if benchmark_values is not None:
            fig.add_trace(go.Scatter(
                x=benchmark_values.index,
                y=benchmark_values.values,
                mode='lines',
                name='Benchmark',
                line=dict(color='red', width=2, dash='dash'),
                hovertemplate='Benchmark: $%{y:,.2f}<br>Date: %{x}<extra></extra>'
            ))
        
        fig.update_layout(
            title='Portfolio Performance',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            height=500,
            hovermode='x unified',
            showlegend=True
        )
        
        return fig
    
    def create_performance_metrics_table(self, metrics: Dict) -> go.Figure:
        """
        Create a table displaying performance metrics.
        
        Args:
            metrics: Dictionary of performance metrics
            
        Returns:
            Plotly figure object (table)
        """
        metric_names = []
        metric_values = []
        
        metric_mapping = {
            'total_return': ('Total Return', f"{metrics.get('total_return', 0):.2f}%"),
            'annualized_return': ('Annualized Return', f"{metrics.get('annualized_return', 0):.2f}%"),
            'annualized_volatility': ('Annualized Volatility', f"{metrics.get('annualized_volatility', 0):.2f}%"),
            'sharpe_ratio': ('Sharpe Ratio', f"{metrics.get('sharpe_ratio', 0):.3f}"),
            'max_drawdown': ('Maximum Drawdown', f"{metrics.get('max_drawdown', 0):.2f}%"),
            'final_value': ('Final Value', f"${metrics.get('final_value', 0):,.2f}")
        }
        
        for key, (name, value) in metric_mapping.items():
            if key in metrics:
                metric_names.append(name)
                metric_values.append(value)
        
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Metric', 'Value'],
                fill_color='lightblue',
                align='center',
                font=dict(size=14, color='black')
            ),
            cells=dict(
                values=[metric_names, metric_values],
                fill_color='white',
                align=['left', 'right'],
                font=dict(size=12)
            )
        )])
        
        fig.update_layout(
            title='Performance Metrics',
            height=300,
            margin=dict(t=50, b=20, l=20, r=20)
        )
        
        return fig
    
    def create_drawdown_chart(self, portfolio_values: pd.Series) -> go.Figure:
        """
        Create a drawdown chart showing portfolio drawdowns over time.
        
        Args:
            portfolio_values: Series of portfolio values
            
        Returns:
            Plotly figure object
        """
        # Calculate drawdowns
        rolling_max = portfolio_values.expanding().max()
        drawdowns = (portfolio_values - rolling_max) / rolling_max * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=drawdowns.index,
            y=drawdowns.values,
            mode='lines',
            fill='tonexty',
            name='Drawdown',
            line=dict(color='red', width=1),
            fillcolor='rgba(255, 0, 0, 0.3)',
            hovertemplate='Drawdown: %{y:.2f}%<br>Date: %{x}<extra></extra>'
        ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        
        fig.update_layout(
            title='Portfolio Drawdown',
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            height=300,
            showlegend=False,
            yaxis=dict(tickformat='.1f')
        )
        
        return fig
    
    def create_sector_allocation_bar(self, sector_weights: Dict[str, float]) -> go.Figure:
        """
        Create a bar chart showing sector allocation.
        
        Args:
            sector_weights: Dictionary mapping sectors to weights
            
        Returns:
            Plotly figure object
        """
        sectors = list(sector_weights.keys())
        weights = [sector_weights[sector] * 100 for sector in sectors]
        
        fig = go.Figure(data=[go.Bar(
            x=sectors,
            y=weights,
            marker_color=self.color_palette[:len(sectors)],
            text=[f"{w:.1f}%" for w in weights],
            textposition='auto'
        )])
        
        fig.update_layout(
            title='Sector Allocation',
            xaxis_title='Sector',
            yaxis_title='Allocation (%)',
            height=400,
            xaxis=dict(tickangle=45)
        )
        
        return fig
    
    def create_risk_return_scatter(self, assets_data: pd.DataFrame) -> go.Figure:
        """
        Create a risk-return scatter plot for individual assets.
        
        Args:
            assets_data: DataFrame with return and volatility data for assets
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=assets_data['volatility'] * 100,
            y=assets_data['expected_return'] * 100,
            mode='markers+text',
            text=assets_data.index,
            textposition='top center',
            marker=dict(
                size=12,
                color=assets_data['sharpe_ratio'],
                colorscale='Viridis',
                colorbar=dict(title="Sharpe Ratio"),
                line=dict(width=1, color='black')
            ),
            name='Assets',
            hovertemplate='%{text}<br>Return: %{y:.2f}%<br>Volatility: %{x:.2f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title='Risk-Return Profile of Assets',
            xaxis_title='Volatility (%)',
            yaxis_title='Expected Return (%)',
            height=500,
            showlegend=False
        )
        
        return fig


class DashboardMetrics:
    """Calculate and format metrics for dashboard display."""
    
    @staticmethod
    def format_percentage(value: float, decimals: int = 2) -> str:
        """Format a decimal value as percentage."""
        return f"{value * 100:.{decimals}f}%"
    
    @staticmethod
    def format_currency(value: float, decimals: int = 2) -> str:
        """Format a value as currency."""
        return f"${value:,.{decimals}f}"
    
    @staticmethod
    def format_number(value: float, decimals: int = 3) -> str:
        """Format a number with specified decimals."""
        return f"{value:.{decimals}f}"
    
    @staticmethod
    def calculate_portfolio_metrics(weights: np.ndarray, tickers: List[str], 
                                   current_prices: Dict[str, float], 
                                   portfolio_value: float) -> Dict:
        """
        Calculate current portfolio metrics.
        
        Args:
            weights: Portfolio weights
            tickers: List of tickers
            current_prices: Current prices dictionary
            portfolio_value: Total portfolio value
            
        Returns:
            Dictionary of portfolio metrics
        """
        metrics = {}
        
        # Calculate position values
        position_values = {}
        total_value = 0
        
        for ticker, weight in zip(tickers, weights):
            if ticker in current_prices:
                position_value = portfolio_value * weight
                position_values[ticker] = position_value
                total_value += position_value
        
        metrics['position_values'] = position_values
        metrics['total_value'] = total_value
        metrics['largest_position'] = max(weights) if len(weights) > 0 else 0
        metrics['number_of_positions'] = len([w for w in weights if w > 0.01])  # Positions > 1%
        
        return metrics


# Create global chart generator instance
chart_generator = ChartGenerator()
dashboard_metrics = DashboardMetrics()