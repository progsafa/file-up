
import os
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, redirect
import pickle

app = Flask(__name__)
model = pickle.load(open("model_no_strict_columns.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    normal_count = attack_count = normal_pct = attack_pct = None

    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            error = "Please upload a CSV file."
        else:
            try:
                df = pd.read_csv(file)
                X = df.select_dtypes(include=["number"])

                # تأكد من وجود نفس عدد الخصائص التي يتوقعها النموذج
                if X.shape[1] != model.n_features_in_:
                    error = f"The uploaded file has {X.shape[1]} features, but the model expects {model.n_features_in_}."
                else:
                    y_pred = model.predict(X)
                    normal_count = (y_pred == "Normal").sum()
                    attack_count = (y_pred == "Attack").sum()
                    total = len(y_pred)
                    normal_pct = round((normal_count / total) * 100, 2)
                    attack_pct = round((attack_count / total) * 100, 2)

                    # رسم الرسم البياني
                    labels = ["Normal", "Attack"]
                    sizes = [normal_count, attack_count]
                    colors = ["green", "red"]
                    plt.figure(figsize=(5, 5))
                    plt.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=140)
                    plt.axis("equal")
                    os.makedirs("static", exist_ok=True)
                    plt.savefig("static/chart.png")
                    plt.close()

            except Exception as e:
                error = f"Error processing file: {e}"

    return render_template("index.html", error=error, normal_count=normal_count,
                           attack_count=attack_count, normal_pct=normal_pct,
                           attack_pct=attack_pct)

if __name__ == "__main__":
    app.run(debug=True)
