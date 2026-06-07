# 🧠 NeuroScanAI

AI-Powered EEG Analysis Platform for Seizure Detection and Speech Decoding

NeuroScanAI is a full-stack healthcare AI platform that enables researchers and clinicians to upload EEG recordings, perform automated seizure analysis, visualize risk timelines, and generate professional PDF reports. The platform also includes support for EEG-based speech decoding workflows.

---

## 🚀 Features

### Authentication & Security

* JWT-based Authentication
* Secure User Registration & Login
* Google OAuth Integration
* Organization-based Access Control

### Seizure Detection

* Upload EDF EEG recordings
* Automated EEG preprocessing
* Feature extraction pipeline
* EEGNet-based deep learning inference
* Seizure risk assessment
* Epoch-wise probability predictions
* Interactive probability timeline visualization

### Speech Decoding

* EEG Speech Decoding Pipeline
* KaraOne Dataset Support
* EEGNet Classification Architecture
* Feature Engineering & Dimensionality Reduction

### Reporting

* Download Analysis Results as JSON
* Generate Professional PDF Reports
* Embedded Seizure Probability Timeline
* Summary Statistics & Risk Assessment

### Dashboard

* Analysis History Tracking
* Status Monitoring
* Risk Visualization
* Multi-user Support

---

# 🏗️ System Architecture

```text
Frontend (React + Vite)
        │
        ▼
FastAPI Backend
        │
 ┌──────┴──────┐
 ▼             ▼
PostgreSQL   ML Inference
 Database      Engine
                 │
        ┌────────┴────────┐
        ▼                 ▼
   Seizure Model    Speech Decoder
     (EEGNet)         (EEGNet)
```

---

# 🧠 Machine Learning Pipeline

## Seizure Detection

### Input

EDF EEG Recordings

### Preprocessing

* Bandpass Filtering
* Channel Selection
* Epoch Generation
* Signal Normalization

### Feature Extraction

* Band Power Features
* Hjorth Parameters
* Spectral Entropy
* Statistical Features

### Model

* EEGNet Deep Learning Architecture
* Binary Classification
* Seizure vs Non-Seizure Detection

### Output

* Risk Level
* Epoch Predictions
* Seizure Probability Timeline
* PDF Report

---

# 🛠️ Tech Stack

## Frontend

* React.js
* Vite
* React Router
* Axios
* Recharts
* Tailwind CSS
* Lucide Icons

## Backend

* FastAPI
* SQLAlchemy
* Async PostgreSQL
* JWT Authentication
* ReportLab
* Matplotlib

## Machine Learning

* PyTorch
* EEGNet
* NumPy
* SciPy
* MNE
* Scikit-Learn
* XGBoost

## Database

* PostgreSQL

---

# 📂 Project Structure

```text
neuroscan/
│
├── backend/
│   ├── auth/
│   ├── db/
│   ├── routers/
│   ├── inference/
│   └── api.py
│
├── frontend/
│   ├── src/
│   ├── components/
│   ├── pages/
│   └── context/
│
├── pipeline/
│   ├── seizure/
│   └── speech/
│
└── README.md
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/Gaurang-Joshi-learner/NeuroScanAI.git
cd NeuroScanAI
```

---

## Backend Setup

```bash
cd backend

python -m venv .venv

source .venv/bin/activate
# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

Create `.env`

```env
DATABASE_URL=postgresql+asyncpg://user:password@localhost/neuroscan

SECRET_KEY=your_secret_key

FRONTEND_URL=http://localhost:5173
```

Run Backend

```bash
uvicorn api:app --reload
```

---

## Frontend Setup

```bash
cd frontend

npm install

npm run dev
```

---

# 📊 Model Performance

## Seizure Detection

| Metric         | Score            |
| -------------- | ---------------- |
| Architecture   | EEGNet           |
| Classification | Binary           |
| Input          | EDF EEG          |
| Output         | Risk Probability |
| Reporting      | PDF + JSON       |

*Performance may vary depending on patient-specific EEG recordings and preprocessing settings.*

---

# 📈 Example Workflow

```text
Upload EDF File
        │
        ▼
Signal Preprocessing
        │
        ▼
Feature Extraction
        │
        ▼
EEGNet Inference
        │
        ▼
Risk Assessment
        │
        ▼
Interactive Dashboard
        │
        ▼
PDF Report Generation
```

---



---

# 🔮 Future Improvements

* Real-time EEG Streaming
* Multi-patient Batch Analysis
* Advanced Deep Learning Architectures
* Clinical Annotation Tools
* Explainable AI Visualizations
* Cloud-based Model Serving
* Role-Based Access Management
* Multi-Organization Collaboration

---




---

