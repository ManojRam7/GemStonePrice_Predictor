from flask import Flask, jsonify, render_template, request

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application


@app.route("/", methods=["GET"])
def home_page():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict_datapoint():
    try:
        data = CustomData(
            carat=float(request.form.get("carat", 0.0)),
            depth=float(request.form.get("depth", 0.0)),
            table=float(request.form.get("table", 0.0)),
            x=float(request.form.get("x", 0.0)),
            y=float(request.form.get("y", 0.0)),
            z=float(request.form.get("z", 0.0)),
            cut=request.form.get("cut", "Ideal"),
            color=request.form.get("color", "D"),
            clarity=request.form.get("clarity", "IF"),
        )
        pred_df = data.get_data_as_dataframe()
        prediction = PredictPipeline().predict(pred_df)
        result = round(float(prediction[0]), 2)
        return render_template("index.html", results=result, pred_df=pred_df.to_dict(orient="records")[0])
    except Exception as exc:
        return render_template("index.html", error=str(exc))


@app.route("/predictAPI", methods=["POST"])
def predict_api():
    payload = request.get_json(force=True)
    data = CustomData(
        carat=float(payload["carat"]),
        depth=float(payload["depth"]),
        table=float(payload["table"]),
        x=float(payload["x"]),
        y=float(payload["y"]),
        z=float(payload["z"]),
        cut=payload["cut"],
        color=payload["color"],
        clarity=payload["clarity"],
    )

    pred_df = data.get_data_as_dataframe()
    prediction = PredictPipeline().predict(pred_df)
    return jsonify({"price": round(float(prediction[0]), 2)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
