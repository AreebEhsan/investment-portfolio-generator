# 📈 Investment Portfolio Recommendation Engine MVP

A comprehensive Streamlit web application that generates optimal investment portfolio allocations using **Modern Portfolio Theory** with real-time market data. Built with Python, featuring interactive visualizations, risk analysis, and automated portfolio optimization.

![Portfolio Optimizer](https://img.shields.io/badge/Portfolio-Optimizer-blue) ![Python](https://img.shields.io/badge/Python-3.8%2B-green) ![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red) ![Alpha Vantage](https://img.shields.io/badge/API-Alpha%20Vantage-orange)

## 🚀 Features

### Core Functionality
- **📊 Modern Portfolio Theory Implementation** - Mean-variance optimization for optimal risk-adjusted returns
- **🎯 Risk-Based Portfolio Allocation** - Conservative, Moderate, and Aggressive profiles
- **📈 Real-Time Market Data** - Live price updates from Alpha Vantage and Yahoo Finance APIs
- **🔄 Automatic Rebalancing** - Portfolio rebalancing recommendations
- **📉 Risk Analysis** - Comprehensive risk metrics including drawdowns, volatility, and correlations

### Optimization Methods
1. **Maximum Sharpe Ratio** - Optimize for best risk-adjusted returns
2. **Minimum Volatility** - Minimize portfolio risk
3. **Risk-Based Allocation** - Tailored to user's risk tolerance

### Interactive Visualizations
- **🥧 Portfolio Allocation Pie Charts** - Visual breakdown of asset weights
- **📈 Performance Charts** - Historical backtesting with benchmark comparison
- **🌡️ Risk Heatmaps** - Asset correlation analysis
- **📊 Efficient Frontier** - Risk-return optimization curves
- **📉 Drawdown Analysis** - Portfolio risk assessment over time

## 🛠️ Technology Stack

- **Frontend:** Streamlit (latest version)
- **Data Analysis:** pandas, numpy, scipy
- **Optimization:** scipy.optimize, cvxpy (optional)
- **Visualization:** Plotly, matplotlib
- **Data Sources:** Alpha Vantage API (primary), Yahoo Finance (backup)
- **Environment Management:** python-dotenv

## 📁 Project Structure

```
investment-portfolio-gen/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── .env                           # Environment variables (your API keys)
├── .env.template                  # Environment template
├── README.md                      # This file
├── src/                           # Source code modules
│   ├── config.py                  # Configuration management
│   ├── data_fetcher.py           # Market data fetching
│   ├── optimizer.py              # Portfolio optimization algorithms  
│   └── dashboard.py              # Chart generation and visualization
├── data/                         # Data storage (auto-created)
├── charts/                       # Chart exports (auto-created)
└── logs/                        # Application logs (auto-created)
```

## ⚡ Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd investment-portfolio-gen

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
# Copy environment template
cp .env.template .env

# Edit .env file with your API keys
```

**Required API Keys (All Free Tier):**

1. **Alpha Vantage** (Primary data source - Free: 25 calls/day)
   - Get your free API key: https://www.alphavantage.co/support/#api-key
   - Add to `.env`: `ALPHA_VANTAGE_API_KEY=your_key_here`

2. **Yahoo Finance** (Backup - Completely Free)
   - No API key needed, works automatically via `yfinance`

**Optional API Keys:**

3. **Finnhub** (Additional data - Free: 60 calls/minute)
   - Get free API key: https://finnhub.io/register
   - Add to `.env`: `FINNHUB_API_KEY=your_key_here`

### 3. Run the Application

```bash
# Start the Streamlit app
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 🎯 How to Use

### Step 1: Configure Your Portfolio
1. **💰 Set Starting Capital** - Enter your investment amount
2. **🎲 Choose Risk Tolerance** - Select Conservative, Moderate, or Aggressive
3. **📅 Set Investment Horizon** - How long you plan to invest
4. **🏢 Select Assets** - Choose from stocks, ETFs, or create custom selection

### Step 2: Optimize Your Portfolio  
1. **⚙️ Choose Optimization Method** - Max Sharpe Ratio, Min Volatility, or Risk-Based
2. **🔧 Adjust Advanced Settings** - Position limits, rebalancing frequency
3. **🚀 Click "Optimize Portfolio"** - Get your personalized recommendation

### Step 3: Analyze Results
1. **📊 Review Allocation** - See your optimized portfolio weights
2. **📈 Check Performance** - Historical backtesting and metrics
3. **🎯 Understand Risk** - Correlation analysis and drawdown charts
4. **📉 View Efficient Frontier** - Optimal risk-return combinations

## 🔧 Configuration Options

### Risk Profiles

| Profile | Description | Max Position | Min Diversification |
|---------|-------------|--------------|-------------------|
| **Conservative** | Low risk, capital preservation | 25% | 8 assets |
| **Moderate** | Balanced risk-return | 35% | 6 assets |  
| **Aggressive** | High growth potential | 50% | 4 assets |

### Data Sources

- **Primary:** Alpha Vantage API (25 free calls/day)
- **Backup:** Yahoo Finance via yfinance (unlimited, free)
- **Rate Limiting:** Automatically managed to respect API limits
- **Caching:** 10-minute cache to optimize API usage

## 🚀 Deployment on Streamlit Cloud

### Prerequisites
- GitHub repository with your code
- Streamlit Cloud account (free at https://share.streamlit.io/)

### Deployment Steps

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Connect to Streamlit Cloud:**
   - Go to https://share.streamlit.io/
   - Click "New app"
   - Connect your GitHub repository
   - Select branch: `main`
   - Main file path: `app.py`

3. **Configure Secrets:**
   - In Streamlit Cloud app settings, go to "Secrets"
   - Add your API keys in TOML format:
   ```toml
   # Streamlit secrets format
   ALPHA_VANTAGE_API_KEY = "your_alpha_vantage_key_here"
   FINNHUB_API_KEY = "your_finnhub_key_here"
   NEWS_API_KEY = "your_news_api_key_here"
   ```

4. **Deploy:**
   - Click "Deploy"
   - Your app will be live at `https://your-app-name.streamlit.app/`

### Alternative Deployment Options

#### Heroku
1. Create `Procfile`:
   ```
   web: sh setup.sh && streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```
2. Create `setup.sh`:
   ```bash
   mkdir -p ~/.streamlit/
   echo "[server]
   headless = true
   port = $PORT
   enableCORS = false
   " > ~/.streamlit/config.toml
   ```

#### Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

## 🔒 Security & API Best Practices

### API Key Management
- ✅ Store API keys in environment variables
- ✅ Never commit API keys to version control
- ✅ Use `.env` file for local development
- ✅ Use platform secrets for deployment
- ✅ Respect rate limits to avoid API key suspension

### Free Tier Limits
- **Alpha Vantage:** 25 requests/day (managed automatically)
- **Yahoo Finance:** Unlimited (no key required)
- **Finnhub:** 60 requests/minute (managed automatically)

## 📊 Example Portfolio Outputs

### Conservative Portfolio (Sample)
```
Asset Allocation:
- VTI (Total Stock Market): 35%
- BND (Total Bond Market): 30%  
- VNQ (REITs): 15%
- VXUS (International): 20%

Expected Annual Return: 7.2%
Annual Volatility: 8.5%
Sharpe Ratio: 0.85
```

### Aggressive Portfolio (Sample)
```
Asset Allocation:
- TSLA: 25%
- NVDA: 20%
- AAPL: 15%
- GOOGL: 15%
- AMZN: 12%
- Others: 13%

Expected Annual Return: 15.8%
Annual Volatility: 22.1%  
Sharpe Ratio: 0.71
```

## 🐛 Troubleshooting

### Common Issues

**1. API Key Errors**
```
Error: Alpha Vantage API key not found
```
**Solution:** Check your `.env` file and ensure `ALPHA_VANTAGE_API_KEY=your_key_here`

**2. Rate Limit Exceeded**
```
Warning: Alpha Vantage daily rate limit reached  
```
**Solution:** App automatically falls back to Yahoo Finance. Wait 24 hours for Alpha Vantage reset.

**3. Optimization Failed**
```
Error: Portfolio optimization failed
```
**Solution:** Try different assets, reduce position constraints, or check data availability.

**4. No Market Data**
```
Error: Could not fetch market data
```
**Solution:** Check internet connection, verify ticker symbols, try different time periods.

### Debug Mode
Enable debug mode in `.env`:
```
APP_DEBUG=True
LOG_LEVEL=DEBUG
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Make your changes and add tests
4. Commit changes: `git commit -am 'Add new feature'`
5. Push to branch: `git push origin feature/new-feature`
6. Submit a Pull Request

## 📈 Roadmap

### Version 1.1 (Planned)
- [ ] Cryptocurrency portfolio support
- [ ] ESG (Environmental, Social, Governance) scoring
- [ ] Monte Carlo simulation for risk assessment
- [ ] Email alerts for rebalancing
- [ ] Portfolio comparison tools

### Version 1.2 (Future)
- [ ] Machine learning predictions
- [ ] Sector rotation strategies
- [ ] Options and derivatives support
- [ ] Multi-currency support
- [ ] Mobile app version

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Modern Portfolio Theory** by Harry Markowitz
- **Alpha Vantage** for providing free market data API
- **Yahoo Finance** for reliable backup data source
- **Streamlit** for the excellent web app framework
- **Plotly** for interactive visualizations

## 📞 Support

- **Issues:** Create a GitHub issue for bugs or feature requests
- **Documentation:** Check this README and inline code comments
- **API Support:** Refer to Alpha Vantage and Yahoo Finance documentation

---

**⚠️ Disclaimer:** This application is for educational and research purposes only. It does not constitute financial advice. Past performance does not guarantee future results. Always consult with qualified financial advisors before making investment decisions.

---

Made with ❤️ and ☕ by [Your Name]