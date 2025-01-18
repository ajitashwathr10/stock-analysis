import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns 
from datetime import datetime, timedelta
import requests
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import logging

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CompanyOverview:
    symbol: str
    name: str
    sector: str
    industry: str
    market_cap: float
    country: str
    employees: int

class StockAnalysis:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.stock = yf.Ticker(self.symbol)
        self.data = {}
    
    def get_company_overview(self) -> CompanyOverview:
        try:
            info = self.stock.info
            return CompanyOverview(
                symbol = self.symbol,
                name = info.get('longName', 'N/A'),
                sector = info.get('sector', 'N/A'),
                industry = info.get('industry', 'N/A'),
                market_cap = info.get('marketCap', 0),
                country = info.get('country', 'N/A'),
                employees = info.get('employees', 0)
            )
        except Exception as e:
            logger.error(f"Error retrieving company overview for {self.symbol}: ")
            return logger
        
    def get_financial_statements(self, period: str = 'yearly') -> Dict:
        try:
            income_stmt = self.stock.financials if period == 'yearly' else self.stock.quarterly_financials
            balance_sheet = self.stock.balance_sheet if period == 'yearly' else self.stock.quarterly_balance_sheet
            cash_flow = self.stock.cashflow if period == 'yearly' else self.stock.quarterly_cashflow
            return {
                'income_statement': income_stmt,
                'balance_sheet': balance_sheet,
                'cash_flow': cash_flow
            }
        except Exception as e:
            logger.error(f"Error retrieving financial statements for {self.symbol}: {str(e)}")
            raise
    
    def calculate_valuation_ratios(self) -> Dict[str, float]:
        try:
            info = self.stock.info
            pe_ratio = info.get('trailingPE', 0)
            pb_ratio = info.get('priceToBook', 0)
            ps_ratio = info.get('priceToSalesTrailing12Months', 0)
            dividend_yield = info.get('dividendYield', 0)
            earning_yield = (1 / pe_ratio * 100) if pe_ratio else 0
            return {
                'pe_ratio': pe_ratio,
                'pb_ratio': pb_ratio,
                'ps_ratio': ps_ratio,
                'dividend_yield': dividend_yield,
                'earning_yield': earning_yield
            }
        except Exception as e:
            logger.error(f"Error calculating valuation ratios for {self.symbol}: {str(e)}")
            raise
    
    def calculate_growth_metrics(self, period: str = 'yearly') -> Dict[str, float]:
        try:
            financials = self.get_financial_statements(period)
            income_stmt = financials['income_statement']
            cash_flow = financials['cash_flow']
        
            revenue_growth = self.calculate_growth_rate(income_stmt.loc['Total Revenue'])
            eps_growth = self.calculate_growth_rate(income_stmt.loc['Net Income'])
            fcf_growth = self.caclulate_growth_rate(cash_flow.loc['Free Cash Flow'])
            return {
                'revenue_growth': revenue_growth,
                'eps_growth': eps_growth,
                'fcf_growth': fcf_growth
            }
        except Exception as e:
            logger.error(f"Error calculating growth metrics for {self.symbol}: {str(e)}")
            raise
    
    def calculate_profitability_ratios(self) -> Dict[str, float]:
        try:
            financials = self.get_financial_statements()
            income_stmt = financials['income_statement']
            balance_sheet = financials['balance_sheet']
            net_income = income_stmt.loc['Net Income'].iloc[0]
            total_assets = balance_sheet.loc['Total Assets'].iloc[0]
            total_equity = balance_sheet.loc['Total Stockholder Equity'].iloc[0]
            
            roa = (net_income / total_assets) * 100
            roe = (net_income / total_equity) * 100

            return {
                'roa': roa,
                'roe': roe,
                'roi': roe
            }
        except Exception as e:
            logger.error(f"Error calculating profitability ratios for {self.symbol}: {str(e)}")
            raise
    
    def calculate_debt_ratios(self) -> Dict[str, float]:
        try:
            financials = self.get_financial_statements()
            balance_sheet = financials['balance_sheet']
            income_stmt = financials['income_statement']

            total_debt = balance_sheet.loc['Total Debt'].iloc[0]
            total_assets = balance_sheet.loc['Total Assets'].iloc[0]
            total_equity = balance_sheet.loc['Total Stockholder Equity'].iloc[0]
            ebit = income_stmt.loc['Operating Income'].iloc[0]
        
            return {
                'debt_to_equity': total_debt / total_equity,
                'debt_to_assets': total_debt / total_assets,
                'interest_coverage': ebit / (total_debt * 0.05)
            }
        except Exception as e:
            logger.error(f"Error calculating debt ratios for {self.symbol}: {str(e)}")
            raise
    
    def calculate_liquidity_ratios(self) -> Dict[str, float]:
        try:
            balance_sheet = self.get_financial_statements()['balance_sheet']
            current_assets = balance_sheet.loc['Total Current Assets'].iloc[0]
            current_liabilities = balance_sheet.loc['Total Current Liabilities'].iloc[0]
            inventory = balance_sheet.loc['Inventory'].iloc[0]

            return {
                'current_ratio': current_assets / current_liabilities,
                'quick_ratio': (current_assets - inventory) / current_liabilities
            }
        except Exception as e:
            logger.error(f"Error calculating liquidity ratios for {self.symbol}: {str(e)}")
            raise
    
    def get_market_sentiment(self, period: str = '1y') -> Dict:
        try:
            hist_data = self.stock.history(period = period)
            hist_data['MA50'] = hist_data['Close'].rolling(window = 50).mean()
            hist_data['MA200'] = hist_data['Close'].rolling(window = 200).mean()
            
            delta = hist_data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window = 14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window = 14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            current_price = hist_data['Close'].iloc[-1]
            
            return {
                'current_price': current_price,
                'ma50': hist_data['MA50'].iloc[-1],
                'ma200': hist_data['MA200'].iloc[-1],
                'rsi': rsi.iloc[-1],
                'price_momentum': (current_price / hist_data['Close'].iloc[0] - 1) * 100
            }
        except Exception as e:
            logger.error(f"Error calculating market sentiment for {self.symbol}: {str(e)}")
            raise

    def compare_with_peers(self, peer_symbols: List[str]) -> pd.DataFrame:
        try:
            metrics = []
            for symbol in [self.symbol] + peer_symbols:
                analyzer = StockAnalysis(symbol)
                valuation = analyzer.calculate_valuation_ratios()
                profitability = analyzer.calculate_profitability_ratios()
                
                metrics.append({
                    'Symbol': symbol,
                    'P/E Ratio': valuation['pe_ratio'],
                    'P/B Ratio': valuation['pb_ratio'],
                    'ROE': profitability['roe'],
                    'ROA': profitability['roa']
                })
            
            return pd.DataFrame(metrics)
        except Exception as e:
            logger.error(f"Error comparing with peers: {str(e)}")
            raise

    def generate_report(self, output_format: str = 'text') -> str:
        try:
            overview = self.get_company_overview()
            valuation = self.calculate_valuation_ratios()
            growth = self.calculate_growth_metrics()
            profitability = self.calculate_profitability_ratios()
            debt = self.calculate_debt_ratios()
            liquidity = self.calculate_liquidity_ratios()
            sentiment = self.get_market_sentiment()
            
            if output_format == 'text':
                report = f"""
Stock Analysis Report for {overview.name} ({overview.symbol})
=============================================

Company Overview:
----------------
Sector: {overview.sector}
Industry: {overview.industry}
Market Cap: ${overview.market_cap:,.2f}
Employees: {overview.employees:,}

Valuation Metrics:
-----------------
P/E Ratio: {valuation['pe_ratio']:.2f}
P/B Ratio: {valuation['pb_ratio']:.2f}
P/S Ratio: {valuation['ps_ratio']:.2f}
Dividend Yield: {valuation['dividend_yield']:.2f}%

Growth Metrics (YoY):
-------------------
Revenue Growth: {growth['revenue_growth']:.2f}%
EPS Growth: {growth['eps_growth']:.2f}%
FCF Growth: {growth['fcf_growth']:.2f}%

Profitability:
-------------
ROA: {profitability['roa']:.2f}%
ROE: {profitability['roe']:.2f}%
ROI: {profitability['roi']:.2f}%

Debt Metrics:
------------
Debt/Equity: {debt['debt_to_equity']:.2f}
Debt/Assets: {debt['debt_to_assets']:.2f}
Interest Coverage: {debt['interest_coverage']:.2f}

Liquidity:
----------
Current Ratio: {liquidity['current_ratio']:.2f}
Quick Ratio: {liquidity['quick_ratio']:.2f}

Market Sentiment:
---------------
Current Price: ${sentiment['current_price']:.2f}
50-day MA: ${sentiment['ma50']:.2f}
200-day MA: ${sentiment['ma200']:.2f}
RSI: {sentiment['rsi']:.2f}
Price Momentum (1Y): {sentiment['price_momentum']:.2f}%
"""
                return report
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
                
        except Exception as e:
            logger.error(f"Error generating report for {self.symbol}: {str(e)}")
            raise

    def _calculate_growth_rate(self, series: pd.Series) -> float:
        try:
            current = series.iloc[0]
            previous = series.iloc[1]
            return ((current - previous) / abs(previous)) * 100
        except Exception:
            return 0

    def plot_key_metrics(self, metrics: List[str], period: str = '1y') -> None:
        try:
            hist_data = self.stock.history(period=period)
            
            plt.figure(figsize = (15, 10))
            for i, metric in enumerate(metrics, 1):
                plt.subplot(len(metrics), 1, i)
                plt.plot(hist_data.index, hist_data[metric])
                plt.title(f'{metric} Over Time')
                plt.grid(True)
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            logger.error(f"Error plotting metrics for {self.symbol}: {str(e)}")
            raise

class DataValidator:
    @staticmethod
    def validate_financial_data(data: pd.DataFrame) -> bool:
        try:
            if data.isnull().sum().sum() > 0:
                logger.warning("Missing values detected in financial data")
                return False
            if (data.select_dtypes(include = [np.number]) < 0).any().any():
                logger.warning("Negative va lues detected in financial data")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Error validating financial data: {str(e)}")
            return False

def main():
    try:
        symbol = input("Enter stock symbol: ")
        analyzer = StockAnalysis(symbol)
        report = analyzer.generate_report()
        print(report)
        
        analyzer.plot_key_metrics(['Close', 'Volume'])

        peers = [] #Mention some stock symbol for comparsion
        peer_comparison = analyzer.compare_with_peers(peers)
        print("\nPeer Comparison:")
        print(peer_comparison)
        
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        raise

if __name__ == "__main__":
    main()

