import joblib
import pandas as pd
from rapidfuzz import process, fuzz


class StudentRiskPredictor:

    def __init__(self, model_path):
        self.cols = [
            "Hours_Studied",
            "Prev_Mid_Sem_Marks",
            "Prev_Sem_Marks",
            "Attendance_%",
            "Mobile_Screen_Time_hrs"
        ]
        self.model = self.load_model(model_path)

    def load_model(self, path):
        try:
            return joblib.load(path)
        except Exception as e:
            return None

    def clean(self, text):
        try:
            return text.strip().lower().replace("_", "").replace(" ", "")
        except:
            return ""

    def match_columns(self, user_cols):
        mapping = {}
        cleaned_standard = [self.clean(c) for c in self.cols]

        for col in user_cols:
            match, score, _ = process.extractOne(
                self.clean(col),
                cleaned_standard,
                scorer=fuzz.token_sort_ratio
            )

            original_match = self.cols[cleaned_standard.index(match)]

            if score > 70:
                mapping[col] = original_match

        return mapping

    def rename_columns(self, df, mapping):
        df.rename(columns=mapping, inplace=True)
        return df

    def predict(self, df):
        if self.model is None:
            return {"error": "Model not loaded"}

        missing_cols = [col for col in self.cols if col not in df.columns]
        if missing_cols:
            return {"error": f"Missing columns: {missing_cols}"}

        new_df = df[self.cols]
        df["prediction"] = self.model.predict(new_df)

        return df