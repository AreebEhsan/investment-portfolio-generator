"""
Investment Portfolio Recommendation Engine - Main Streamlit Application
A comprehensive portfolio optimization tool using Modern Portfolio Theory with real-time market data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import asyncio
from loguru import logger

# Import our modules
from src.config import config, Config
from src.data_fetcher import data_fetcher
from src.optimizer import portfolio_optimizer, RiskProfileManager
from src.dashboard import chart_generator, dashboard_metrics

# Page configuration
st.set_page_config(
    page_title="Investment Portfolio Recommendation Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-card {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-card {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stSelectbox > div > div {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'portfolio_data' not in st.session_state:
        st.session_state.portfolio_data = None
    if 'optimization_result' not in st.session_state:
        st.session_state.optimization_result = None
    if 'current_prices' not in st.session_state:
        st.session_state.current_prices = {}
    if 'last_update' not in st.session_state:
        st.session_state.last_update = datetime.now()
    if 'selected_tickers' not in st.session_state:
        st.session_state.selected_tickers = config.DEFAULT_STOCKS[:5]


def main():
    """Main application function."""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">📈 Investment Portfolio Recommendation Engine</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem; color: #666;">
        Optimize your investment portfolio using Modern Portfolio Theory with real-time market data
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for user inputs
    with st.sidebar:
        st.header("🎯 Portfolio Configuration")
        
        # Show API status
        config.show_api_status()
        
        st.markdown("---")
        
        # Investment parameters
        st.subheader("Investment Parameters")
        
        portfolio_value = st.number_input(
            "💰 Starting Capital ($)",
            min_value=1000,
            max_value=10000000,
            value=int(config.DEFAULT_PORTFOLIO_SIZE),
            step=1000,
            help="Total amount to invest"
        )
        
        risk_profile = st.selectbox(
            "🎲 Risk Tolerance",
            options=['Conservative', 'Moderate', 'Aggressive'],
            index=1,
            help="Your risk tolerance level affects portfolio allocation constraints"
        )
        
        st.write(f"**Risk Profile:** {RiskProfileManager.get_profile_description(risk_profile)}")
        
        investment_horizon = st.slider(
            "📅 Investment Horizon (Years)",
            min_value=1,
            max_value=30,
            value=5,
            help="How long you plan to hold the portfolio"
        )
        
        st.markdown("---")
        
        # Asset selection
        st.subheader("🏢 Asset Selection")
        
        asset_type = st.radio(
            "Asset Categories",
            ["Stocks", "ETFs", "Mixed", "Custom"],
            index=0
        )
        
        if asset_type == "Stocks":
            available_assets = config.DEFAULT_STOCKS
        elif asset_type == "ETFs":
            available_assets = config.DEFAULT_ETFS
        elif asset_type == "Mixed":
            available_assets = config.DEFAULT_STOCKS + config.DEFAULT_ETFS
        else:  # Custom
            available_assets = []
        
        if asset_type != "Custom":
            # Get valid defaults that exist in available_assets
            valid_defaults = [ticker for ticker in st.session_state.selected_tickers if ticker in available_assets]
            if not valid_defaults and available_assets:
                valid_defaults = available_assets[:5]  # Default to first 5 assets
            
            selected_tickers = st.multiselect(
                "Select Assets",
                options=available_assets,
                default=valid_defaults,
                help="Choose 3-15 assets for optimal diversification"
            )
        else:
            custom_tickers = st.text_input(
                "Enter Tickers (comma-separated)",
                value=",".join(st.session_state.selected_tickers),
                help="e.g., AAPL,MSFT,GOOGL,TSLA"
            )
            selected_tickers = [ticker.strip().upper() for ticker in custom_tickers.split(",") if ticker.strip()]
        
        # Sector preferences (optional)
        st.subheader("🏭 Sector Preferences (Optional)")
        preferred_sectors = st.multiselect(
            "Preferred Sectors",
            options=list(config.SECTOR_TICKERS.keys()),
            help="Leave empty for no sector preference"
        )
        
        if preferred_sectors:
            sector_tickers = []
            for sector in preferred_sectors:
                sector_tickers.extend(config.SECTOR_TICKERS[sector])
            # Add sector tickers to selection
            selected_tickers = list(set(selected_tickers + sector_tickers))
        
        st.markdown("---")
        
        # Optimization settings
        st.subheader("⚙️ Optimization Settings")
        
        optimization_method = st.selectbox(
            "Optimization Method",
            ["Max Sharpe Ratio", "Min Volatility", "Risk-Based Allocation"],
            index=0,
            help="Choose the optimization objective"
        )
        
        # Advanced settings in expander
        with st.expander("Advanced Settings"):
            max_position_size = st.slider(
                "Max Single Position (%)",
                min_value=5,
                max_value=50,
                value=30,
                help="Maximum weight for any single asset"
            ) / 100
            
            min_position_size = st.slider(
                "Min Position Size (%)",
                min_value=0,
                max_value=10,
                value=2,
                help="Minimum weight for included assets"
            ) / 100
            
            rebalance_frequency = st.selectbox(
                "Rebalancing Frequency",
                ["Monthly", "Quarterly", "Semi-Annual", "Annual"],
                index=1
            )
        
        st.markdown("---")
        
        # Action buttons
        if st.button("🚀 Optimize Portfolio", type="primary", use_container_width=True):
            if len(selected_tickers) < 3:
                st.error("Please select at least 3 assets for diversification.")
            elif len(selected_tickers) > 20:
                st.error("Please select no more than 20 assets to avoid over-diversification.")
            else:
                with st.spinner("Optimizing portfolio... This may take a moment."):
                    optimize_portfolio(selected_tickers, portfolio_value, risk_profile, 
                                    optimization_method, max_position_size, min_position_size)
        
        if st.button("📊 Update Prices", use_container_width=True):
            with st.spinner("Fetching latest prices..."):
                update_current_prices(selected_tickers)
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox(
            "🔄 Auto-refresh prices",
            value=False,
            help=f"Refresh every {config.UPDATE_INTERVAL_SECONDS} seconds"
        )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.session_state.optimization_result is not None:
            display_optimization_results()
        else:
            display_welcome_screen()
    
    with col2:
        if st.session_state.current_prices:
            display_current_prices()
        
        if st.session_state.portfolio_data is not None:
            display_portfolio_metrics()
    
    # Auto-refresh functionality
    if auto_refresh:
        time.sleep(config.UPDATE_INTERVAL_SECONDS)
        st.rerun()


def optimize_portfolio(tickers: List[str], portfolio_value: float, risk_profile: str,
                      optimization_method: str, max_position: float, min_position: float):
    """Optimize the portfolio based on user parameters."""
    try:
        # Validate tickers
        valid_tickers = data_fetcher.validate_tickers(tickers)
        if not valid_tickers:
            st.error("No valid tickers found. Please check your selections.")
            return
        
        if len(valid_tickers) < len(tickers):
            st.warning(f"Some tickers were invalid. Using {len(valid_tickers)} valid tickers: {', '.join(valid_tickers)}")
        
        # Fetch historical data
        with st.status("Fetching market data...") as status:
            price_data = data_fetcher.get_portfolio_data(valid_tickers, period="1y")
            
            if price_data.empty:
                st.error("Could not fetch market data. Please try again or select different assets.")
                return
            
            status.update(label="Calculating returns and metrics...", state="running")
            
            # Calculate returns and metrics
            returns = portfolio_optimizer.calculate_returns(price_data)
            metrics = portfolio_optimizer.calculate_metrics(returns)
            
            status.update(label="Optimizing portfolio...", state="running")
            
            # Set up constraints
            constraints = {
                'max_weight': max_position,
                'min_weight': min_position
            }
            
            # Optimize based on selected method
            if optimization_method == "Max Sharpe Ratio":
                result = portfolio_optimizer.optimize_max_sharpe(
                    metrics['expected_returns'].values,
                    metrics['cov_matrix'].values,
                    constraints
                )
            elif optimization_method == "Min Volatility":
                result = portfolio_optimizer.optimize_min_volatility(
                    metrics['expected_returns'].values,
                    metrics['cov_matrix'].values,
                    constraints
                )
            else:  # Risk-Based Allocation
                result = portfolio_optimizer.risk_based_allocation(
                    metrics['expected_returns'].values,
                    metrics['cov_matrix'].values,
                    risk_profile.lower()
                )
            
            status.update(label="Portfolio optimization complete!", state="complete")
        
        if result['optimization_success']:
            # Store results in session state
            st.session_state.optimization_result = result
            st.session_state.optimization_result['tickers'] = valid_tickers
            st.session_state.optimization_result['portfolio_value'] = portfolio_value
            st.session_state.optimization_result['risk_profile'] = risk_profile
            st.session_state.optimization_result['metrics'] = metrics
            st.session_state.portfolio_data = price_data
            st.session_state.selected_tickers = valid_tickers
            
            # Get current prices
            update_current_prices(valid_tickers)
            
            st.success("✅ Portfolio optimization completed successfully!")
        else:
            st.error("❌ Portfolio optimization failed. Please try different parameters.")
            
    except Exception as e:
        logger.error(f"Error in portfolio optimization: {e}")
        st.error(f"An error occurred during optimization: {str(e)}")


def update_current_prices(tickers: List[str]):
    """Update current prices for selected tickers."""
    try:
        prices = data_fetcher.get_current_prices(tickers)
        st.session_state.current_prices = prices
        st.session_state.last_update = datetime.now()
        
        if prices:
            st.success(f"✅ Updated prices for {len(prices)} assets")
        else:
            st.warning("⚠️ Could not fetch current prices")
            
    except Exception as e:
        logger.error(f"Error updating prices: {e}")
        st.error(f"Error updating prices: {str(e)}")


def display_welcome_screen():
    """Display welcome screen with instructions."""
    st.markdown("""
    ## Welcome to Your Portfolio Optimizer! 🎯
    
    Get started by configuring your investment parameters in the sidebar:
    
    1. **💰 Set your starting capital** - How much you want to invest
    2. **🎲 Choose your risk tolerance** - Conservative, Moderate, or Aggressive
    3. **📅 Select investment horizon** - How long you plan to invest
    4. **🏢 Pick your assets** - Choose from stocks, ETFs, or create a custom mix
    5. **🚀 Click "Optimize Portfolio"** to get your personalized recommendation
    
    ### 🔧 Features:
    - **Modern Portfolio Theory** optimization using mean-variance analysis
    - **Real-time market data** from Alpha Vantage and Yahoo Finance
    - **Interactive charts** and visualizations
    - **Risk-return analysis** with efficient frontier
    - **Backtesting** and performance metrics
    - **Automatic rebalancing** recommendations
    
    ### 📊 Data Sources:
    - **Primary:** Alpha Vantage API (free tier)
    - **Backup:** Yahoo Finance (completely free)
    - **Rate limits:** Automatically managed to respect free tier limits
    """)
    
    # Display some market stats or example
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="💹 Market Coverage",
            value="10,000+",
            help="Stocks and ETFs available"
        )
    
    with col2:
        st.metric(
            label="🎯 Optimization Methods",
            value="3",
            help="Max Sharpe, Min Volatility, Risk-Based"
        )
    
    with col3:
        st.metric(
            label="📈 Risk Profiles",
            value="3",
            help="Conservative, Moderate, Aggressive"
        )


def display_optimization_results():
    """Display the portfolio optimization results."""
    result = st.session_state.optimization_result
    
    st.header("🎯 Optimized Portfolio Results")
    
    # Portfolio metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Expected Annual Return",
            f"{result['expected_return']*100:.2f}%",
            help="Expected portfolio return based on historical data"
        )
    
    with col2:
        st.metric(
            "Annual Volatility",
            f"{result['volatility']*100:.2f}%",
            help="Expected portfolio risk (standard deviation)"
        )
    
    with col3:
        st.metric(
            "Sharpe Ratio",
            f"{result['sharpe_ratio']:.3f}",
            help="Risk-adjusted return metric (higher is better)"
        )
    
    with col4:
        st.metric(
            "Portfolio Value",
            f"${result['portfolio_value']:,.2f}",
            help="Total portfolio value"
        )
    
    # Portfolio allocation chart
    st.subheader("📊 Portfolio Allocation")
    
    allocation_fig = chart_generator.create_portfolio_allocation_pie(
        result['weights'],
        result['tickers'],
        f"Optimized {result['risk_profile']} Portfolio"
    )
    st.plotly_chart(allocation_fig, use_container_width=True)
    
    # Detailed allocation table
    allocation_df = pd.DataFrame({
        'Ticker': result['tickers'],
        'Weight': [f"{w*100:.2f}%" for w in result['weights']],
        'Value': [f"${result['portfolio_value'] * w:,.2f}" for w in result['weights']]
    })
    allocation_df = allocation_df.sort_values('Weight', ascending=False)
    
    st.subheader("💼 Detailed Allocation")
    st.dataframe(allocation_df, use_container_width=True)
    
    # Performance analysis
    if st.session_state.portfolio_data is not None:
        st.subheader("📈 Historical Performance Analysis")
        
        # Backtest the portfolio
        backtest_results = portfolio_optimizer.backtest_portfolio(
            result['weights'],
            st.session_state.portfolio_data,
            result['portfolio_value']
        )
        
        # Performance chart
        performance_fig = chart_generator.create_portfolio_value_chart(
            backtest_results['portfolio_values']
        )
        st.plotly_chart(performance_fig, use_container_width=True)
        
        # Performance metrics table
        metrics_fig = chart_generator.create_performance_metrics_table(backtest_results)
        st.plotly_chart(metrics_fig, use_container_width=True)
        
        # Additional analysis in tabs
        tab1, tab2, tab3 = st.tabs(["Risk Analysis", "Asset Correlation", "Efficient Frontier"])
        
        with tab1:
            # Risk analysis
            drawdown_fig = chart_generator.create_drawdown_chart(backtest_results['portfolio_values'])
            st.plotly_chart(drawdown_fig, use_container_width=True)
            
            # Risk metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Maximum Drawdown",
                    f"{backtest_results['max_drawdown']:.2f}%",
                    help="Largest peak-to-trough decline"
                )
            with col2:
                st.metric(
                    "Sharpe Ratio",
                    f"{backtest_results['sharpe_ratio']:.3f}",
                    help="Risk-adjusted return"
                )
        
        with tab2:
            # Correlation analysis
            correlation_fig = chart_generator.create_correlation_heatmap(
                result['metrics']['corr_matrix']
            )
            st.plotly_chart(correlation_fig, use_container_width=True)
            
            st.write("**Interpretation:**")
            st.write("- Values close to 1 indicate high positive correlation")
            st.write("- Values close to -1 indicate high negative correlation") 
            st.write("- Values close to 0 indicate low correlation")
        
        with tab3:
            # Efficient frontier
            try:
                frontier_data = portfolio_optimizer.generate_efficient_frontier(
                    result['metrics']['expected_returns'].values,
                    result['metrics']['cov_matrix'].values
                )
                
                if not frontier_data.empty:
                    frontier_fig = chart_generator.create_efficient_frontier(
                        frontier_data, result
                    )
                    st.plotly_chart(frontier_fig, use_container_width=True)
                    
                    st.write("**Efficient Frontier:** Shows optimal risk-return combinations")
                    st.write("**Red Star:** Your optimized portfolio")
                else:
                    st.warning("Could not generate efficient frontier data")
            except Exception as e:
                st.warning(f"Could not generate efficient frontier: {str(e)}")


def display_current_prices():
    """Display current prices sidebar."""
    st.subheader("💹 Current Prices")
    
    if st.session_state.current_prices:
        prices_data = []
        for ticker, price in st.session_state.current_prices.items():
            prices_data.append({
                'Ticker': ticker,
                'Price': f"${price:.2f}" if price else "N/A"
            })
        
        prices_df = pd.DataFrame(prices_data)
        st.dataframe(prices_df, use_container_width=True, hide_index=True)
        
        st.caption(f"Last updated: {st.session_state.last_update.strftime('%H:%M:%S')}")
    else:
        st.info("Click 'Update Prices' to fetch current market data")


def display_portfolio_metrics():
    """Display additional portfolio metrics."""
    if st.session_state.optimization_result is None:
        return
    
    st.subheader("📊 Portfolio Metrics")
    
    result = st.session_state.optimization_result
    
    # Calculate additional metrics
    weights = result['weights']
    tickers = result['tickers']
    
    # Concentration metrics
    largest_position = max(weights) * 100
    positions_over_5pct = len([w for w in weights if w > 0.05])
    effective_positions = 1 / sum(w**2 for w in weights)  # Effective number of positions
    
    st.metric(
        "Largest Position",
        f"{largest_position:.1f}%",
        help="Size of largest single position"
    )
    
    st.metric(
        "Positions > 5%",
        f"{positions_over_5pct}",
        help="Number of significant positions"
    )
    
    st.metric(
        "Effective Positions",
        f"{effective_positions:.1f}",
        help="Effective number of positions (accounts for concentration)"
    )
    
    # Risk profile info
    st.info(f"**Risk Profile:** {result['risk_profile']}")
    st.caption(RiskProfileManager.get_profile_description(result['risk_profile']))


if __name__ == "__main__":
    main()