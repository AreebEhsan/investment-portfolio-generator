"""
Portfolio optimization module implementing Modern Portfolio Theory.
Uses mean-variance optimization to find optimal portfolio allocations.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
import streamlit as st
from loguru import logger

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    logger.warning("CVXPY not available, using scipy optimization only")

from .config import config


class PortfolioOptimizer:
    """Portfolio optimizer using Modern Portfolio Theory."""
    
    def __init__(self, risk_free_rate: Optional[float] = None):
        """
        Initialize portfolio optimizer.
        
        Args:
            risk_free_rate: Risk-free rate for Sharpe ratio calculation
        """
        self.risk_free_rate = risk_free_rate or config.RISK_FREE_RATE
    
    def calculate_returns(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate daily returns from price data.
        
        Args:
            price_data: DataFrame with price data
            
        Returns:
            DataFrame with daily returns
        """
        returns = price_data.pct_change().dropna()
        return returns
    
    def calculate_metrics(self, returns: pd.DataFrame) -> Dict:
        """
        Calculate portfolio metrics from returns data.
        
        Args:
            returns: DataFrame with daily returns
            
        Returns:
            Dictionary containing expected returns, covariance matrix, and other metrics
        """
        # Annualized expected returns (assuming 252 trading days)
        expected_returns = returns.mean() * 252
        
        # Covariance matrix (annualized)
        cov_matrix = returns.cov() * 252
        
        # Correlation matrix
        corr_matrix = returns.corr()
        
        # Volatility (annualized standard deviation)
        volatility = returns.std() * np.sqrt(252)
        
        # Sharpe ratios for individual assets
        sharpe_ratios = (expected_returns - self.risk_free_rate) / volatility
        
        return {
            'expected_returns': expected_returns,
            'cov_matrix': cov_matrix,
            'corr_matrix': corr_matrix,
            'volatility': volatility,
            'sharpe_ratios': sharpe_ratios,
            'num_assets': len(returns.columns)
        }
    
    def portfolio_performance(self, weights: np.ndarray, expected_returns: np.ndarray, 
                            cov_matrix: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate portfolio performance metrics.
        
        Args:
            weights: Portfolio weights
            expected_returns: Expected returns array
            cov_matrix: Covariance matrix
            
        Returns:
            Tuple of (expected_return, volatility, sharpe_ratio)
        """
        portfolio_return = np.sum(expected_returns * weights)
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        return portfolio_return, portfolio_volatility, sharpe_ratio
    
    def optimize_max_sharpe(self, expected_returns: np.ndarray, cov_matrix: np.ndarray,
                           constraints: Optional[Dict] = None) -> Dict:
        """
        Optimize portfolio for maximum Sharpe ratio.
        
        Args:
            expected_returns: Expected returns array
            cov_matrix: Covariance matrix
            constraints: Optional constraints dictionary
            
        Returns:
            Optimization result dictionary
        """
        num_assets = len(expected_returns)
        
        def negative_sharpe(weights):
            portfolio_return, portfolio_volatility, sharpe_ratio = self.portfolio_performance(
                weights, expected_returns, cov_matrix
            )
            return -sharpe_ratio
        
        # Constraints
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
        
        # Add custom constraints if provided
        if constraints:
            if 'max_weight' in constraints:
                max_weight = constraints['max_weight']
                cons.append({'type': 'ineq', 'fun': lambda x: max_weight - x})
            
            if 'min_weight' in constraints:
                min_weight = constraints['min_weight']
                cons.append({'type': 'ineq', 'fun': lambda x: x - min_weight})
        
        # Bounds for each weight (0 to 1, or custom if provided)
        bounds = constraints.get('bounds', tuple((0, 1) for _ in range(num_assets)))
        
        # Initial guess (equal weights)
        x0 = np.array([1/num_assets] * num_assets)
        
        # Optimize
        result = minimize(
            negative_sharpe,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000}
        )
        
        if result.success:
            optimal_weights = result.x
            portfolio_return, portfolio_volatility, sharpe_ratio = self.portfolio_performance(
                optimal_weights, expected_returns, cov_matrix
            )
            
            return {
                'weights': optimal_weights,
                'expected_return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio,
                'optimization_success': True,
                'method': 'max_sharpe'
            }
        else:
            logger.error(f"Optimization failed: {result.message}")
            return {
                'weights': x0,
                'expected_return': 0,
                'volatility': 0,
                'sharpe_ratio': 0,
                'optimization_success': False,
                'method': 'max_sharpe'
            }
    
    def optimize_min_volatility(self, expected_returns: np.ndarray, cov_matrix: np.ndarray,
                               constraints: Optional[Dict] = None) -> Dict:
        """
        Optimize portfolio for minimum volatility.
        
        Args:
            expected_returns: Expected returns array
            cov_matrix: Covariance matrix
            constraints: Optional constraints dictionary
            
        Returns:
            Optimization result dictionary
        """
        num_assets = len(expected_returns)
        
        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # Constraints
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
        
        # Add custom constraints if provided
        if constraints:
            if 'max_weight' in constraints:
                max_weight = constraints['max_weight']
                cons.append({'type': 'ineq', 'fun': lambda x: max_weight - x})
            
            if 'min_weight' in constraints:
                min_weight = constraints['min_weight']
                cons.append({'type': 'ineq', 'fun': lambda x: x - min_weight})
        
        # Bounds for each weight
        bounds = constraints.get('bounds', tuple((0, 1) for _ in range(num_assets)))
        
        # Initial guess
        x0 = np.array([1/num_assets] * num_assets)
        
        # Optimize
        result = minimize(
            portfolio_volatility,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000}
        )
        
        if result.success:
            optimal_weights = result.x
            portfolio_return, portfolio_volatility, sharpe_ratio = self.portfolio_performance(
                optimal_weights, expected_returns, cov_matrix
            )
            
            return {
                'weights': optimal_weights,
                'expected_return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio,
                'optimization_success': True,
                'method': 'min_volatility'
            }
        else:
            logger.error(f"Optimization failed: {result.message}")
            return {
                'weights': x0,
                'expected_return': 0,
                'volatility': 0,
                'sharpe_ratio': 0,
                'optimization_success': False,
                'method': 'min_volatility'
            }
    
    def optimize_target_return(self, expected_returns: np.ndarray, cov_matrix: np.ndarray,
                              target_return: float, constraints: Optional[Dict] = None) -> Dict:
        """
        Optimize portfolio for a target return with minimum volatility.
        
        Args:
            expected_returns: Expected returns array
            cov_matrix: Covariance matrix
            target_return: Target portfolio return
            constraints: Optional constraints dictionary
            
        Returns:
            Optimization result dictionary
        """
        num_assets = len(expected_returns)
        
        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        # Constraints
        cons = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
            {'type': 'eq', 'fun': lambda x: np.sum(expected_returns * x) - target_return}  # Target return
        ]
        
        # Add custom constraints if provided
        if constraints:
            if 'max_weight' in constraints:
                max_weight = constraints['max_weight']
                cons.append({'type': 'ineq', 'fun': lambda x: max_weight - x})
            
            if 'min_weight' in constraints:
                min_weight = constraints['min_weight']
                cons.append({'type': 'ineq', 'fun': lambda x: x - min_weight})
        
        # Bounds for each weight
        bounds = constraints.get('bounds', tuple((0, 1) for _ in range(num_assets)))
        
        # Initial guess
        x0 = np.array([1/num_assets] * num_assets)
        
        # Optimize
        result = minimize(
            portfolio_volatility,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000}
        )
        
        if result.success:
            optimal_weights = result.x
            portfolio_return, portfolio_volatility, sharpe_ratio = self.portfolio_performance(
                optimal_weights, expected_returns, cov_matrix
            )
            
            return {
                'weights': optimal_weights,
                'expected_return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio,
                'optimization_success': True,
                'method': 'target_return'
            }
        else:
            logger.warning(f"Target return optimization failed: {result.message}")
            # Fallback to max Sharpe optimization
            return self.optimize_max_sharpe(expected_returns, cov_matrix, constraints)
    
    def risk_based_allocation(self, expected_returns: np.ndarray, cov_matrix: np.ndarray,
                             risk_tolerance: str) -> Dict:
        """
        Create risk-based portfolio allocation.
        
        Args:
            expected_returns: Expected returns array
            cov_matrix: Covariance matrix
            risk_tolerance: Risk tolerance level ('conservative', 'moderate', 'aggressive')
            
        Returns:
            Portfolio allocation dictionary
        """
        if risk_tolerance.lower() == 'conservative':
            # Minimize volatility with some diversification constraints
            constraints = {'max_weight': 0.3, 'min_weight': 0.01}
            return self.optimize_min_volatility(expected_returns, cov_matrix, constraints)
        
        elif risk_tolerance.lower() == 'aggressive':
            # Maximize Sharpe ratio with concentrated positions allowed
            constraints = {'max_weight': 0.6, 'min_weight': 0.0}
            return self.optimize_max_sharpe(expected_returns, cov_matrix, constraints)
        
        else:  # moderate
            # Balance between return and risk
            constraints = {'max_weight': 0.4, 'min_weight': 0.02}
            return self.optimize_max_sharpe(expected_returns, cov_matrix, constraints)
    
    def generate_efficient_frontier(self, expected_returns: np.ndarray, cov_matrix: np.ndarray,
                                   num_portfolios: int = 50) -> pd.DataFrame:
        """
        Generate efficient frontier data points.
        
        Args:
            expected_returns: Expected returns array
            cov_matrix: Covariance matrix
            num_portfolios: Number of portfolios to generate
            
        Returns:
            DataFrame with efficient frontier data
        """
        # Calculate range of target returns
        min_ret = expected_returns.min()
        max_ret = expected_returns.max()
        target_returns = np.linspace(min_ret, max_ret, num_portfolios)
        
        efficient_portfolios = []
        
        for target_ret in target_returns:
            try:
                result = self.optimize_target_return(expected_returns, cov_matrix, target_ret)
                if result['optimization_success']:
                    efficient_portfolios.append({
                        'return': result['expected_return'],
                        'volatility': result['volatility'],
                        'sharpe_ratio': result['sharpe_ratio'],
                        'weights': result['weights']
                    })
            except Exception as e:
                logger.warning(f"Failed to optimize for return {target_ret}: {e}")
                continue
        
        return pd.DataFrame(efficient_portfolios)
    
    def backtest_portfolio(self, weights: np.ndarray, price_data: pd.DataFrame,
                          initial_value: float = 10000) -> Dict:
        """
        Backtest portfolio performance.
        
        Args:
            weights: Portfolio weights
            price_data: Historical price data
            initial_value: Initial portfolio value
            
        Returns:
            Backtest results dictionary
        """
        returns = self.calculate_returns(price_data)
        
        # Calculate portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1)
        
        # Calculate cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod()
        portfolio_values = cumulative_returns * initial_value
        
        # Calculate performance metrics
        total_return = (portfolio_values.iloc[-1] / initial_value - 1) * 100
        annualized_return = (portfolio_values.iloc[-1] / initial_value) ** (252 / len(portfolio_returns)) - 1
        annualized_volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - self.risk_free_rate) / annualized_volatility
        
        # Calculate maximum drawdown
        rolling_max = portfolio_values.expanding().max()
        drawdowns = (portfolio_values - rolling_max) / rolling_max
        max_drawdown = drawdowns.min() * 100
        
        return {
            'portfolio_values': portfolio_values,
            'portfolio_returns': portfolio_returns,
            'cumulative_returns': cumulative_returns,
            'total_return': total_return,
            'annualized_return': annualized_return * 100,
            'annualized_volatility': annualized_volatility * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_value': portfolio_values.iloc[-1]
        }


class RiskProfileManager:
    """Manage different risk profiles and their characteristics."""
    
    RISK_PROFILES = {
        'Conservative': {
            'description': 'Low risk, stable returns, capital preservation',
            'target_volatility': 0.08,
            'max_single_position': 0.25,
            'min_diversification': 8,
            'bond_allocation': 0.4,
            'equity_allocation': 0.6
        },
        'Moderate': {
            'description': 'Balanced risk-return, moderate growth',
            'target_volatility': 0.12,
            'max_single_position': 0.35,
            'min_diversification': 6,
            'bond_allocation': 0.2,
            'equity_allocation': 0.8
        },
        'Aggressive': {
            'description': 'High risk, high growth potential',
            'target_volatility': 0.18,
            'max_single_position': 0.5,
            'min_diversification': 4,
            'bond_allocation': 0.1,
            'equity_allocation': 0.9
        }
    }
    
    @classmethod
    def get_profile_constraints(cls, profile: str) -> Dict:
        """Get optimization constraints for a risk profile."""
        if profile not in cls.RISK_PROFILES:
            profile = 'Moderate'  # Default fallback
        
        profile_data = cls.RISK_PROFILES[profile]
        
        return {
            'max_weight': profile_data['max_single_position'],
            'min_weight': 1 / profile_data['min_diversification'] / 2,  # Half of equal weight
            'target_volatility': profile_data['target_volatility']
        }
    
    @classmethod
    def get_profile_description(cls, profile: str) -> str:
        """Get description for a risk profile."""
        return cls.RISK_PROFILES.get(profile, cls.RISK_PROFILES['Moderate'])['description']


# Create global optimizer instance
portfolio_optimizer = PortfolioOptimizer()