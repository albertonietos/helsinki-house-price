# 🏠 Helsinki House Price Analysis

A comprehensive analysis and prediction system for property prices in the Helsinki metropolitan area, featuring data collection, exploratory analysis, and an interactive multi-page Streamlit application.

## 🌐 Live Demo

I've built an interactive demo to explore the results:
**[🔗 Live Application](https://albertonietos-helsinki-house-apphelsinki-house-price-app-494vpl.streamlit.app/)**

## 📚 Research & Analysis

This project includes comprehensive Jupyter notebooks covering the entire data science pipeline:

- **[📊 Web Scraping](https://github.com/albertonietos/helsinki-house-price/blob/main/notebooks/01-web-scrapper-notebook.ipynb)** - Data collection from Etuovi.com
- **[🧹 Data Cleaning](https://github.com/albertonietos/helsinki-house-price/blob/main/notebooks/02-data-cleaning-notebook.ipynb)** - Raw data preprocessing and validation
- **[🔍 Exploratory Analysis](https://github.com/albertonietos/helsinki-house-price/blob/main/notebooks/03-exploratory-data-analysis.ipynb)** - Statistical analysis and geographical visualization

## 🚀 Interactive Application Features

### 📊 Data Explorer
- Interactive property map with multiple color schemes (percentile-based, logarithmic, price tiers)
- 3D density visualization showing property distribution
- Geographic analysis across Helsinki, Vantaa, and Espoo
- Real-time property statistics and insights

### 🔮 Price Predictor
- Machine learning-powered price predictions using Random Forest
- 25+ location presets covering the Helsinki metropolitan area
- Market comparison and contextual analysis
- Property characteristic impact visualization

### 🧠 Model Analysis
- Comprehensive performance metrics (R², MAE, RMSE, MAPE)
- Interactive feature importance analysis with error bars
- Partial dependence plots showing individual feature effects
- Model reliability assessment and insights

## 📁 Project Structure

```
helsinki-house-price/
├── app/
│   ├── 🏠_Home.py                      # Main landing page
│   └── pages/
│       ├── 1_📊_Data_Explorer.py       # Data visualization and exploration
│       ├── 2_🔮_Price_Predictor.py     # Price prediction interface
│       └── 3_🧠_Model_Analysis.py      # Model performance and analysis
├── notebooks/                          # Jupyter analysis notebooks
├── data/
│   └── cleaned/
│       └── helsinki_house_price_cleaned.xls
├── requirements.txt
└── README.md
```

## 🛠️ Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/albertonietos/helsinki-house-price.git
   cd helsinki-house-price
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Running the Application

```bash
streamlit run app/🏠_Home.py
```

The application will open in your browser at `http://localhost:8501`

## 📊 Data Source

The dataset contains property listings scraped from [Etuovi.com](https://www.etuovi.com/) covering:
- **Helsinki** - City center and surrounding areas
- **Vantaa** - Airport region and suburbs
- **Espoo** - Tech hub and residential areas

### Features Used
- **Size**: Property area in square meters
- **Year**: Year the property was built
- **Total_rooms**: Number of rooms
- **Latitude/Longitude**: Geographic coordinates
- **Price**: Target variable (in euros)

## 🤖 Machine Learning Model

- **Algorithm**: Random Forest Regressor
- **Features**: Size, Year, Total number of rooms, Latitude, Longitude
- **Performance**: ~80% R² score
- **Validation**: Train/test split

## 🎨 Visualization Features

### Color Schemes
- **Percentile-based**: Balanced color distribution
- **Logarithmic**: Good for wide price ranges
- **Price tiers**: Quartile-based categories
- **Linear**: Traditional min-max scaling

### Interactive Elements
- Hover tooltips with property details
- Zoom and pan capabilities
- Real-time parameter updates
- Responsive design

## 📈 Model Insights

The analysis reveals key factors affecting Helsinki property prices:

1. **Size** - Most important factor (~60% importance)
2. **Location** - Latitude/Longitude combined (~25% importance)
3. **Year Built** - Age and condition factor (~10% importance)
4. **Room Count** - Layout and functionality (~5% importance)

## ⚠️ Limitations

- Model based on limited feature set
- Market conditions change over time
- External factors not captured (schools, transport, etc.)
- **Remember**: *All models are wrong, some models are useful*

## 🔧 Technical Details

### Dependencies
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms
- **Plotly**: Interactive visualizations
- **PyDeck**: Advanced mapping capabilities

### Performance Optimizations
- `@st.cache_data` for data loading
- `@st.cache_resource` for model training
- Efficient data processing pipelines
- Responsive UI design

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**Alberto Nieto**
- LinkedIn: [albertonietosandino](https://www.linkedin.com/in/albertonietosandino/)
- GitHub: [Your GitHub Profile]

## 🙏 Acknowledgments

- Data sourced from [Etuovi.com](https://www.etuovi.com/)
- Built with [Streamlit](https://streamlit.io/)
- Visualizations powered by [Plotly](https://plotly.com/) and [PyDeck](https://pydeck.gl/)

---

*For questions, suggestions, or issues, please open a GitHub issue or contact me.*
