# Supervised Learning

This folder contains supervised learning projects that demonstrate various machine learning techniques using labeled datasets. These projects cover time-series prediction, regression, and classification tasks using both traditional machine learning algorithms and deep learning approaches.

## 📊 Projects Overview

### 1. [Stock Price Prediction with LSTM](./project1.ipynb)
**Goal**: Use an LSTM neural network to predict the closing price of a stock using its past data.

**Key Features**:
- Historical stock data from Yahoo Finance (AAPL)
- LSTM architecture with multiple layers and dropout
- Time-series sequence creation (180-day lookback)
- Data normalization using MinMaxScaler
- Hyperparameter tuning with grid search
- Performance evaluation and visualization

**Technologies**: TensorFlow/Keras, LSTM, yfinance, scikit-learn, pandas

**Architecture**:
- 3-layer LSTM with 50-75 units per layer
- Dropout layers (0.2-0.3) for regularization
- Dense output layer for price prediction
- Adam optimizer with mean squared error loss

**Performance**: Achieved low validation loss with effective price trend prediction

### 2. [Bike Rental Prediction with Time-Series Models](./project2.ipynb)
**Goal**: Predict daily bike rental counts using time-series modeling (ARIMA), based on historical rental data.

**Key Features**:
- Time-series data preprocessing and analysis
- ARIMA model implementation with pmdarima
- Seasonal decomposition and trend analysis
- Model validation and forecasting
- Performance metrics and visualization

**Technologies**: pmdarima, pandas, matplotlib, statsmodels, scikit-learn

**Model Components**:
- Auto-ARIMA for parameter selection
- Seasonal ARIMA (SARIMA) for seasonal patterns
- Time-series cross-validation
- Forecast accuracy evaluation

### 3. [Project 3](./project3.ipynb)
**Goal**: Additional supervised learning implementation (details to be explored).

**Key Features**: TBD based on notebook content

### 4. [Project 4](./project4.ipynb)
**Goal**: Additional supervised learning implementation (details to be explored).

**Key Features**: TBD based on notebook content

### 5. [Project 5](./project5.ipynb)
**Goal**: Additional supervised learning implementation (details to be explored).

**Key Features**: TBD based on notebook content

## 🛠️ Common Technologies Used

- **Deep Learning**: TensorFlow, Keras, LSTM networks
- **Time-Series**: pmdarima, statsmodels, ARIMA models
- **Data Processing**: pandas, NumPy, scikit-learn
- **Visualization**: matplotlib, seaborn
- **Data Sources**: yfinance, custom datasets
- **Model Evaluation**: Cross-validation, performance metrics

## 🚀 Getting Started

### Prerequisites
```bash
pip install tensorflow pandas numpy matplotlib scikit-learn
pip install yfinance  # For stock data
pip install pmdarima  # For ARIMA models
pip install statsmodels  # For time-series analysis
```

### Running the Projects

1. **Choose a project** from the list above
2. **Open the notebook** in Jupyter or Google Colab
3. **Install dependencies** as specified in each notebook
4. **Download datasets** (stock data, bike rental data)
5. **Follow the step-by-step implementation**
6. **Experiment** with different parameters and models

### Google Colab Integration
Most notebooks include direct links to run in Google Colab:
- Click the "Open In Colab" badge at the top of each notebook
- Enable GPU for faster LSTM training
- Some projects may require specific library versions

## 📈 Key Concepts Covered

### Time-Series Analysis
- **Data Preprocessing**: Handling missing values, outliers
- **Feature Engineering**: Creating lag features, rolling statistics
- **Seasonality**: Identifying and modeling seasonal patterns
- **Trend Analysis**: Detecting and removing trends
- **Stationarity**: Making time series stationary for modeling

### LSTM Networks
- **Architecture**: Long Short-Term Memory cells for sequence modeling
- **Sequence Creation**: Converting time series to supervised learning format
- **Hyperparameter Tuning**: Optimizing network architecture
- **Regularization**: Dropout and other techniques to prevent overfitting
- **Performance Evaluation**: Metrics for time-series prediction

### ARIMA Models
- **Auto-ARIMA**: Automatic parameter selection
- **Seasonal ARIMA**: Handling seasonal patterns
- **Model Diagnostics**: Checking model assumptions
- **Forecasting**: Making future predictions
- **Validation**: Time-series cross-validation techniques

## 🎯 Learning Objectives

After completing these projects, you will understand:

- **Time-Series Fundamentals**: Understanding temporal data patterns
- **LSTM Networks**: Building and training recurrent neural networks
- **ARIMA Modeling**: Classical time-series forecasting techniques
- **Data Preprocessing**: Preparing time-series data for modeling
- **Model Evaluation**: Assessing prediction performance
- **Hyperparameter Tuning**: Optimizing model parameters
- **Feature Engineering**: Creating meaningful features from time series

## 🔧 Model Architectures

### LSTM Architecture
```
Input Layer (180 time steps, 1 feature)
↓
LSTM Layer 1 (50-75 units, return_sequences=True)
↓
Dropout Layer (0.2-0.3)
↓
LSTM Layer 2 (50-75 units, return_sequences=True)
↓
Dropout Layer (0.2-0.3)
↓
LSTM Layer 3 (50-75 units)
↓
Dropout Layer (0.2-0.3)
↓
Dense Output Layer (1 unit)
```

### ARIMA Model
- **Auto-ARIMA**: Automatic (p,d,q) parameter selection
- **Seasonal Component**: (P,D,Q,s) for seasonal patterns
- **Model Validation**: AIC/BIC criteria for model selection

## 📊 Datasets Used

- **Stock Data**: AAPL historical prices from Yahoo Finance
- **Bike Rental Data**: Daily rental counts with temporal patterns
- **Time-Series Features**: Date, price, volume, technical indicators

## 📈 Performance Optimization

Each project includes techniques for:
- **Data Normalization**: MinMaxScaler for LSTM inputs
- **Sequence Length**: Optimizing lookback periods
- **Model Architecture**: Balancing complexity and performance
- **Regularization**: Dropout and early stopping
- **Hyperparameter Tuning**: Grid search and validation

## 🎨 Applications

These supervised learning techniques can be applied to:

- **Finance**: Stock price prediction, risk assessment
- **Transportation**: Demand forecasting, route optimization
- **Retail**: Sales forecasting, inventory management
- **Energy**: Power consumption prediction, renewable energy forecasting
- **Healthcare**: Patient monitoring, disease progression prediction
- **Marketing**: Customer behavior prediction, campaign optimization

## 📚 Evaluation Metrics

### Time-Series Metrics
- **Mean Absolute Error (MAE)**: Average absolute prediction error
- **Mean Squared Error (MSE)**: Average squared prediction error
- **Root Mean Squared Error (RMSE)**: Square root of MSE
- **Mean Absolute Percentage Error (MAPE)**: Percentage-based error
- **R-squared**: Coefficient of determination

### Model Validation
- **Time-Series Cross-Validation**: Rolling window validation
- **Walk-Forward Analysis**: Sequential validation approach
- **Holdout Validation**: Train/test split for time series

## 🤝 Contributing

Feel free to:
- Add new supervised learning projects
- Improve existing implementations
- Share performance optimizations
- Report issues or suggest enhancements
- Contribute new datasets or use cases

## 📖 Additional Resources

- [TensorFlow Time Series Guide](https://www.tensorflow.org/tutorials/structured_data/time_series) - Official tutorials
- [ARIMA Documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html) - Statsmodels
- [LSTM Paper](https://www.bioinf.jku.at/publications/older/2604.pdf) - Original LSTM research
- [Time Series Analysis](https://otexts.com/fpp2/) - Forecasting principles and practice

## 🔍 Troubleshooting

Common issues and solutions:
- **Memory Issues**: Reduce sequence length or batch size
- **Convergence**: Adjust learning rate and optimizer
- **Overfitting**: Increase dropout or reduce model complexity
- **Data Quality**: Handle missing values and outliers properly

## 🚀 Advanced Topics

For those looking to extend these projects:
- **Multi-variate Time Series**: Multiple input features
- **Ensemble Methods**: Combining multiple models
- **Attention Mechanisms**: Improving LSTM performance
- **Real-time Prediction**: Streaming data processing

---

**Note**: Start with the stock price prediction project to understand LSTM fundamentals, then move to ARIMA models for classical time-series analysis. Each project demonstrates different approaches to supervised learning with temporal data.
