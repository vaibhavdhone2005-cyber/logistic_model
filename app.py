import os
import pickle
import numpy as np
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Load model safely with relative path resolution for serverless environments
MODEL_PATH = os.path.join(os.path.dirname(__file__), "logistic_model.pkl")

model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

# HTML Template with Unreal Dark Glassmorphism & Neon Glow Effects
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Employee Retention Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #070913;
            --card-bg: rgba(18, 22, 41, 0.65);
            --card-border: rgba(255, 255, 255, 0.12);
            --neon-cyan: #00f2fe;
            --neon-purple: #4facfe;
            --accent-pink: #ff0844;
            --accent-violet: #ffb199;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --input-bg: rgba(10, 14, 29, 0.8);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        body {
            background-color: var(--bg-dark);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem 1rem;
            overflow-x: hidden;
            position: relative;
        }

        /* Unreal Ambient Background Orbs */
        .orb {
            position: absolute;
            border-radius: 50%;
            filter: blur(120px);
            z-index: 0;
            opacity: 0.5;
            animation: float 12s ease-in-out infinite alternate;
        }

        .orb-1 {
            width: 450px;
            height: 450px;
            background: radial-gradient(circle, #7f00ff, #e100ff);
            top: -10%;
            left: -10%;
        }

        .orb-2 {
            width: 500px;
            height: 500px;
            background: radial-gradient(circle, #00f2fe, #4facfe);
            bottom: -15%;
            right: -10%;
            animation-delay: -6s;
        }

        @keyframes float {
            0% { transform: translate(0, 0) scale(1); }
            100% { transform: translate(40px, 50px) scale(1.1); }
        }

        .container {
            width: 100%;
            max-width: 900px;
            background: var(--card-bg);
            backdrop-filter: blur(24px);
            -webkit-backdrop-filter: blur(24px);
            border: 1px solid var(--card-border);
            border-radius: 28px;
            padding: 3rem;
            position: relative;
            z-index: 10;
            /* Unreal Multi-layered Shadow Effect */
            box-shadow: 
                0 0 0 1px rgba(255, 255, 255, 0.08) inset,
                0 20px 50px rgba(0, 0, 0, 0.6),
                0 0 80px rgba(0, 242, 254, 0.15),
                0 0 120px rgba(127, 0, 255, 0.12);
            transition: transform 0.4s ease, box-shadow 0.4s ease;
        }

        .container:hover {
            box-shadow: 
                0 0 0 1px rgba(255, 255, 255, 0.15) inset,
                0 30px 70px rgba(0, 0, 0, 0.7),
                0 0 100px rgba(0, 242, 254, 0.25),
                0 0 140px rgba(127, 0, 255, 0.2);
        }

        header {
            text-align: center;
            margin-bottom: 2.5rem;
        }

        header h1 {
            font-size: 2.4rem;
            font-weight: 800;
            background: linear-gradient(135deg, #ffffff 0%, #00f2fe 50%, #4facfe 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
            margin-bottom: 0.5rem;
            text-shadow: 0 10px 20px rgba(0, 242, 254, 0.2);
        }

        header p {
            color: var(--text-secondary);
            font-size: 1rem;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 1.5rem;
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        label {
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: #cbd5e1;
            display: flex;
            align-items: center;
            gap: 0.4rem;
        }

        input, select {
            width: 100%;
            padding: 0.9rem 1.1rem;
            background: var(--input-bg);
            border: 1px solid var(--card-border);
            border-radius: 14px;
            color: #ffffff;
            font-size: 0.95rem;
            outline: none;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.4);
        }

        input:focus, select:focus {
            border-color: var(--neon-cyan);
            box-shadow: 
                inset 0 2px 4px rgba(0, 0, 0, 0.4),
                0 0 0 3px rgba(0, 242, 254, 0.25),
                0 0 20px rgba(0, 242, 254, 0.2);
            transform: translateY(-1px);
        }

        option {
            background-color: #0d111e;
            color: #ffffff;
        }

        .btn-submit {
            grid-column: 1 / -1;
            margin-top: 1.5rem;
            padding: 1.1rem;
            font-size: 1.05rem;
            font-weight: 700;
            border: none;
            border-radius: 16px;
            background: linear-gradient(135deg, #00f2fe 0%, #4facfe 50%, #6b11ff 100%);
            color: #ffffff;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 
                0 10px 30px rgba(0, 242, 254, 0.3),
                0 4px 12px rgba(0, 0, 0, 0.5);
            position: relative;
            overflow: hidden;
        }

        .btn-submit:hover {
            transform: translateY(-2px) scale(1.01);
            box-shadow: 
                0 15px 40px rgba(0, 242, 254, 0.5),
                0 6px 18px rgba(0, 0, 0, 0.6);
        }

        .btn-submit:active {
            transform: translateY(0);
        }

        /* Result Card Overlay */
        .result-card {
            margin-top: 2.5rem;
            padding: 1.8rem;
            border-radius: 20px;
            text-align: center;
            display: none;
            animation: slideUp 0.5s ease forwards;
            position: relative;
            overflow: hidden;
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .result-stay {
            background: rgba(16, 185, 129, 0.12);
            border: 1px solid rgba(16, 185, 129, 0.4);
            box-shadow: 0 0 40px rgba(16, 185, 129, 0.25);
            color: #34d399;
        }

        .result-leave {
            background: rgba(239, 68, 68, 0.12);
            border: 1px solid rgba(239, 68, 68, 0.4);
            box-shadow: 0 0 40px rgba(239, 68, 68, 0.25);
            color: #f87171;
        }

        .result-card h2 {
            font-size: 1.6rem;
            margin-bottom: 0.5rem;
            font-weight: 800;
        }

        .result-card p {
            color: var(--text-secondary);
            font-size: 0.95rem;
        }
    </style>
</head>
<body>

    <div class="orb orb-1"></div>
    <div class="orb orb-2"></div>

    <div class="container">
        <header>
            <h1>Employee Retention AI</h1>
            <p>Predict employee turnover probability using Machine Learning</p>
        </header>

        <form id="predictionForm">
            <div class="grid">
                <div class="form-group">
                    <label>Education</label>
                    <select name="Education" required>
                        <option value="0">Bachelors</option>
                        <option value="1">Masters</option>
                        <option value="2">PHD</option>
                    </select>
                </div>

                <div class="form-group">
                    <label>Joining Year</label>
                    <input type="number" name="JoiningYear" min="2000" max="2026" value="2018" required>
                </div>

                <div class="form-group">
                    <label>City</label>
                    <select name="City" required>
                        <option value="0">Bangalore</option>
                        <option value="1">Pune</option>
                        <option value="2">New Delhi</option>
                    </select>
                </div>

                <div class="form-group">
                    <label>Payment Tier</label>
                    <select name="PaymentTier" required>
                        <option value="1">Tier 1 (Highest)</option>
                        <option value="2">Tier 2 (Medium)</option>
                        <option value="3" selected>Tier 3 (Standard)</option>
                    </select>
                </div>

                <div class="form-group">
                    <label>Age</label>
                    <input type="number" name="Age" min="18" max="70" value="28" required>
                </div>

                <div class="form-group">
                    <label>Gender</label>
                    <select name="Gender" required>
                        <option value="0">Male</option>
                        <option value="1">Female</option>
                    </select>
                </div>

                <div class="form-group">
                    <label>Ever Benched?</label>
                    <select name="EverBenched" required>
                        <option value="0">No</option>
                        <option value="1">Yes</option>
                    </select>
                </div>

                <div class="form-group">
                    <label>Domain Experience (Years)</label>
                    <input type="number" name="ExperienceInCurrentDomain" min="0" max="20" value="3" required>
                </div>

                <button type="submit" class="btn-submit">⚡ Run AI Prediction</button>
            </div>
        </form>

        <div id="resultCard" class="result-card">
            <h2 id="resultTitle">Prediction Result</h2>
            <p id="resultDesc">Analyzing features...</p>
        </div>
    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const formData = new FormData(e.target);
            const data = Object.fromEntries(formData.entries());

            const resultCard = document.getElementById('resultCard');
            const resultTitle = document.getElementById('resultTitle');
            const resultDesc = document.getElementById('resultDesc');

            resultCard.style.display = 'block';
            resultCard.className = 'result-card';
            resultTitle.textContent = 'Processing...';
            resultDesc.textContent = 'Executing inference model...';

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                });

                const res = await response.json();

                if (res.error) {
                    resultTitle.textContent = 'Error';
                    resultDesc.textContent = res.error;
                    return;
                }

                if (res.prediction === 1) {
                    resultCard.className = 'result-card result-leave';
                    resultTitle.textContent = '⚠️ High Risk of Leaving';
                    resultDesc.textContent = `Probability: ${(res.probability * 100).toFixed(1)}% likelihood of attrition.`;
                } else {
                    resultCard.className = 'result-card result-stay';
                    resultTitle.textContent = '✅ Likely to Stay';
                    resultDesc.textContent = `Probability: ${((1 - res.probability) * 100).toFixed(1)}% likelihood of remaining.`;
                }
            } catch (err) {
                resultTitle.textContent = 'Server Error';
                resultDesc.textContent = 'Unable to connect to prediction backend.';
            }
        });
    </script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "logistic_model.pkl not found on server."}), 500

    try:
        data = request.get_json()

        # Extract features in exact order expected by feature_names_in_
        # 1. Education, 2. JoiningYear, 3. City, 4. PaymentTier, 5. Age, 6. Gender, 7. EverBenched, 8. ExperienceInCurrentDomain
        features = [
            float(data.get("Education", 0)),
            float(data.get("JoiningYear", 2018)),
            float(data.get("City", 0)),
            float(data.get("PaymentTier", 3)),
            float(data.get("Age", 28)),
            float(data.get("Gender", 0)),
            float(data.get("EverBenched", 0)),
            float(data.get("ExperienceInCurrentDomain", 3))
        ]

        input_array = np.array([features])
        prediction = int(model.predict(input_array)[0])
        probabilities = model.predict_proba(input_array)[0]
        prob_leave = float(probabilities[1]) if len(probabilities) > 1 else float(prediction)

        return jsonify({
            "prediction": prediction,
            "probability": prob_leave
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
