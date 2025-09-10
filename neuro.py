import pandas as pd
import numpy as np
import customtkinter as ctk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from tkinter import messagebox

# ✅ Set Theme
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ✅ Generate dataset
data = {
    "HeartRate": np.random.randint(60, 120, 300),
    "HRV": np.random.randint(20, 100, 300),
    "SleepScore": np.random.randint(50, 100, 300),
    "GaitStability": np.random.randint(30, 90, 300),
    "ActivityLevel": np.random.randint(40, 100, 300),
    "SpO2": np.random.randint(85, 100, 300),
    "StressLevel": np.random.randint(10, 70, 300),
    "TremorScore": np.random.randint(0, 10, 300),
    "Label": np.random.choice([0, 1, 2], 300)
}

df = pd.DataFrame(data)
features = list(data.keys())[:-1]
y = df["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    df[features], y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

label_mapping = {0: "Normal", 1: "Alzheimer's", 2: "Parkinson's"}


class DiseaseApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("🧠 Disease Prediction System")
        self.geometry("1200x800")
        self.resizable(False, False)

        # ✅ Normal Background
        self.configure(bg="#1c1c1c")  # Solid dark background

        # ✅ Center Frame
        self.center_frame = ctk.CTkFrame(self, fg_color="#2a2a2a",
                                         corner_radius=20, width=500, height=500)
        self.center_frame.place(relx=0.5, rely=0.5, anchor="center")

        # ✅ Title
        self.title_label = ctk.CTkLabel(self.center_frame, text="🩺 Disease Predictor",
                                        font=("Arial", 26, "bold"), text_color="cyan")
        self.title_label.pack(pady=15)

        # ✅ Input Fields
        self.entries = {}
        for field in features:
            row_frame = ctk.CTkFrame(self.center_frame, fg_color="transparent")
            row_frame.pack(pady=5)

            label = ctk.CTkLabel(row_frame, text=f"{field}:", font=("Arial", 16))
            label.pack(side="left", padx=10)

            entry = ctk.CTkEntry(row_frame, font=("Arial", 16),
                                 width=200, fg_color="#202020", text_color="white")
            entry.pack(side="right", padx=10)
            self.entries[field] = entry

        # ✅ Buttons
        btn_frame = ctk.CTkFrame(self.center_frame, fg_color="transparent")
        btn_frame.pack(pady=20)

        self.generate_btn = ctk.CTkButton(btn_frame, text="🎲 Generate Random",
                                          command=self.generate_random, fg_color="blue",
                                          text_color="white", width=180, height=40)
        self.generate_btn.pack(side="left", padx=10)

        self.predict_btn = ctk.CTkButton(btn_frame, text="🔮 Predict",
                                         command=self.predict, fg_color="green",
                                         text_color="white", width=180, height=40)
        self.predict_btn.pack(side="right", padx=10)

        # ✅ Results
        result_frame = ctk.CTkFrame(self.center_frame, fg_color="transparent")
        result_frame.pack(pady=20)

        self.result_label = ctk.CTkLabel(result_frame, text="Disease: -",
                                         font=("Arial", 18), text_color="white")
        self.result_label.grid(row=0, column=0, padx=10)

        self.severity_label = ctk.CTkLabel(result_frame, text="Severity: -",
                                           font=("Arial", 18), text_color="white")
        self.severity_label.grid(row=0, column=1, padx=10)

        self.progressbar = ctk.CTkProgressBar(self.center_frame, width=400, height=20)
        self.progressbar.pack(pady=10)
        self.progressbar.set(0)

        self.percentage_label = ctk.CTkLabel(self.center_frame, text="Confidence: 0%",
                                             font=("Arial", 16), text_color="yellow")
        self.percentage_label.pack(pady=5)

        self.generate_random()

    # ✅ Random Data
    def generate_random(self):
        self.random_data = {field: np.random.randint(50, 120) for field in features}
        self.random_data["SpO2"] = np.random.randint(85, 100)
        self.random_data["TremorScore"] = np.random.randint(0, 10)

        for field, value in self.random_data.items():
            self.entries[field].delete(0, "end")
            self.entries[field].insert(0, str(value))

    # ✅ Prediction
    def predict(self):
        try:
            input_data = {field: float(self.entries[field].get()) for field in features}
            new_df = pd.DataFrame([input_data])

            probabilities = model.predict_proba(new_df)[0] * 100
            prediction_index = model.predict(new_df)[0]
            prediction = label_mapping[prediction_index]
            percentage = max(probabilities)

            if prediction == "Alzheimer's":
                severity, color = "⚠️ High Risk", "red"
            elif prediction == "Parkinson's":
                severity, color = "⚠️ Moderate Risk", "orange"
            else:
                severity, color = "✅ Low Risk", "green"

            self.result_label.configure(text=f"Disease: {prediction}", text_color=color)
            self.severity_label.configure(text=f"Severity: {severity}", text_color=color)

            self.animate_progress(percentage)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    # ✅ Progress Bar Animation
    def animate_progress(self, target_percentage):
        current = self.progressbar.get() * 100
        if current < target_percentage:
            current += 1
            self.progressbar.set(current / 100)
            self.percentage_label.configure(text=f"Confidence: {int(current)}%")
            self.after(15, lambda: self.animate_progress(target_percentage))
        else:
            self.progressbar.set(target_percentage / 100)
            self.percentage_label.configure(text=f"Confidence: {int(target_percentage)}%")


if __name__ == "__main__":
    app = DiseaseApp()
    app.mainloop()
