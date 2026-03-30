from fastapi import FastAPI, UploadFile, File, Form
import pandas as pd
import io
from model import StudentRiskPredictor

app = FastAPI(title="Student Risk Assessment📝")

# ✅ load model once
predictor = StudentRiskPredictor("student_pipe")


@app.get("/")
def home():
    return {"message": "Student Risk API Running"}


@app.post("/predict")
async def predict_api(
    file: UploadFile = File(...),
    columns: str = Form(...)
):
    try:
        # read file
        contents = await file.read()
        file_obj = io.StringIO(contents.decode("utf-8"))
        df = pd.read_csv(file_obj)

        # user columns
        user_cols = [col.strip() for col in columns.split(",")]

        # mapping + rename
        mapping = predictor.match_columns(user_cols)
        df = predictor.rename_columns(df, mapping)

        # prediction
        result = predictor.predict(df)

        if isinstance(result, dict):  # error case
            return result

        return {
            "message": "Success",
            "data": result.to_dict(orient="records")
        }

    except Exception as e:
        return {"error": str(e)}