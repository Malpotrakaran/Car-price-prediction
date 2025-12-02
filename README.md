# 🚗 Car Price Prediction Web App

A Streamlit-based web application for predicting car resale prices using a trained machine learning model.

## Features

- 📊 **Data Insights**: Visualize car price distributions and brand comparisons
- 🧠 **Model Info**: Learn about the Random Forest model used
- 🚗 **Price Prediction**: Enter car details to predict selling price

## Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Malpotrakaran/Car-price-prediction.git
   cd Car-price-prediction
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

   The app will open at `http://localhost:8501`

## Deployment Options

### Option 1: Deploy on Streamlit Cloud (Recommended)

1. Push your code to GitHub
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Sign in with your GitHub account
4. Click "New app" → Select your repository
5. Enter the main file as `app.py`
6. Click "Deploy"

### Option 2: Deploy on Heroku

1. Install Heroku CLI
2. Login to Heroku:
   ```bash
   heroku login
   ```

3. Create a new Heroku app:
   ```bash
   heroku create your-app-name
   ```

4. Push to Heroku:
   ```bash
   git push heroku main
   ```

### Option 3: Deploy on Railway

1. Go to [https://railway.app](https://railway.app)
2. Sign in with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repository
5. Set the start command as: `streamlit run app.py`
6. Deploy

### Option 4: Deploy on Render

1. Go to [https://render.com](https://render.com)
2. Create a new Web Service
3. Connect your GitHub repo
4. Set the build command: `pip install -r requirements.txt`
5. Set the start command: `streamlit run app.py`
6. Deploy

## Requirements

- Python 3.9+
- See `requirements.txt` for all dependencies

## Project Structure

```
Car-price-prediction/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── Procfile                        # Heroku deployment config
├── runtime.txt                     # Python version for Heroku
├── setup.sh                        # Setup script for Heroku
├── .streamlit/config.toml          # Streamlit configuration
├── .gitignore                      # Git ignore rules
├── Car Sell Dataset.csv            # Training dataset
├── carsell_model.pkl               # Trained model file
└── carsale.ipynb                   # Model training notebook
```

## Model Details

- **Algorithm**: Random Forest Regressor (from the notebook)
- **Target**: Log of Selling Price
- **Features**: Brand, Model Name, Model Variant, Car Type, Transmission, Fuel Type, Year, Kilometers, Owner, State, Accidental
- **Performance**:
  - MAE: ₹2.9 Lakh
  - RMSE: ₹6.6 Lakh

## Troubleshooting

### "Dataset not found" error
- Ensure `Car Sell Dataset.csv` is in the same directory as `app.py`
- Check file permissions

### "Model file not found" error
- Ensure `carsell_model.pkl` is in the same directory as `app.py`
- Check file permissions

### Deployment not working
- Ensure all files (`app.py`, `Car Sell Dataset.csv`, `carsell_model.pkl`) are committed to Git
- Check that `requirements.txt` has all dependencies
- Verify deployment platform logs for specific errors

## License

This project is open source and available under the MIT License.

## Author

Karan Malpotrakaran

---

**Note**: The model file and dataset files must be in the same directory as `app.py` for the application to work properly.
