# 💰 Tech Salary Predictor

A machine learning web app that predicts tech job salaries in India based on your profile — built with Python, Scikit-learn, and Streamlit.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)

---

## 🌐 Live Demo
👉 [Click here to try the app](https://your-app-link.streamlit.app)

---

## 📸 Screenshot
<!-- Add screenshot here after deploying -->

---

## ✨ Features
- 🔮 Predict annual salary based on job title, experience, education, location, and skills
- 📊 Shows monthly, annual, and experience bonus breakdown
- 🛠 Supports 6 job roles, 7 locations, 3 education levels, and 12 skills
- ⚡ Random Forest ML model trained on 1000+ data points
- 🎨 Clean, interactive Streamlit UI

---

## 🛠 Tech Stack
| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas & NumPy | Data manipulation |
| Scikit-learn | ML model (Random Forest Regressor) |
| Streamlit | Web app frontend |
| Pickle | Model serialization |

---

## 📁 Project Structure
```
salary-predictor/
├── app.py              # Main Streamlit app
├── train_model.py      # Model training script
├── requirements.txt    # Python dependencies
├── .gitignore          # Git ignore rules
└── README.md           # Project documentation
```

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/cyber-boop1/salary-predictor.git
cd salary-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```
The app will open at `http://localhost:8501` in your browser.

---

## 🤖 How the ML Model Works
1. **Data** — Synthetic dataset of 1000 tech professionals generated with realistic salary patterns
2. **Features** — Job title, years of experience, education, location, skill count, company size
3. **Model** — Random Forest Regressor (100 estimators)
4. **Output** — Predicted annual salary in INR

---

## 📈 Model Performance
| Metric | Value |
|---|---|
| Algorithm | Random Forest Regressor |
| Training samples | 1000 |
| Features used | 6 |
| Salary range | ₹3,00,000 – ₹50,00,000 |

---

## 🔮 Future Improvements
- [ ] Add real-world salary dataset (Glassdoor / Kaggle)
- [ ] Add data visualizations (salary trends by role)
- [ ] Add model accuracy metrics dashboard
- [ ] Support for more cities and roles

---

## 👩‍💻 Author
**Shalini Jaiswal**
- GitHub: [@cyber-boop1](https://github.com/cyber-boop1)
- LinkedIn: [jaiswal-shalini](https://linkedin.com/in/jaiswal-shalini)

---
⭐ If you found this useful, please give it a star!
