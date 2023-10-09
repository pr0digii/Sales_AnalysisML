from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn

app = FastAPI()

# Load model artifacts
artifacts = joblib.load('models/predictive/prediction_artefacts.joblib')
model = artifacts['model']

# Load forecasting model
forecasting_model = joblib.load('models/forecastive/holt_winters_model.joblib')

class PredictionInput(BaseModel):
    Sales_Day: float
    event_type: float
    Daily_volume_of_sales: float
    Daily_selling_price: float
    Day_of_Week: float
    Month: float
    Year: float
    event_name_Christmas: float
    event_name_Cinco_De_Mayo: float
    event_name_Cinco_De_Mayo_OrthodoxEaster: float
    event_name_ColumbusDay: float
    event_name_Easter: float
    event_name_Easter_OrthodoxEaster: float
    event_name_Eid_al_Fitr: float
    event_name_EidAlAdha: float
    event_name_Fathers_day: float
    event_name_Fathers_day_NBAFinalsEnd: float
    event_name_Halloween: float
    event_name_IndependenceDay: float
    event_name_LaborDay: float
    event_name_LentStart: float
    event_name_LentWeek2: float
    event_name_MartinLutherKingDay: float
    event_name_MemorialDay: float
    event_name_Mothers_day: float
    event_name_NBAFinalsEnd: float
    event_name_NBAFinalsStart: float
    event_name_NewYear: float
    event_name_OrthodoxChristmas: float
    event_name_OrthodoxEaster: float
    event_name_Pesach_End: float
    event_name_PresidentsDay: float
    event_name_Purim_End: float
    event_name_Ramadan_starts: float
    event_name_StPatricksDay: float
    event_name_SuperBowl: float
    event_name_Thanksgiving: float
    event_name_ValentinesDay: float
    event_name_VeteransDay: float
    event_name_no_specific_event: float

    
class ForecastInput(BaseModel):
    start_date: str
    end_date: str
    
@app.post("/predict/")
def predict(data: PredictionInput):
    # Convert the data to the expected format
    input_data = [getattr(data, feature) for feature in artifacts['features']]
    prediction = model.predict([input_data])
    return {"prediction": float(prediction[0])}

@app.post("/forecast/")
def forecast_sales(data: ForecastInput):
    # Assuming you forecast for 7 days (change this as needed)
    num_days = 7
    # Extract start and end dates from user input
    start_date = data.start_date
    end_date = data.end_date
    # Implement your forecasting logic here based on the provided date range and forecasting model
    # Replace the following with your actual forecasting code
    loaded_forecast = forecasting_model.forecast(steps=num_days)
    # Return the forecast as JSON
    return {"forecast": loaded_forecast.tolist()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)