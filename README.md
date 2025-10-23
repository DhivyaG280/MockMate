# MockMate: AI-Based Mock Interview System

## Overview

MockMate is an AI-powered mock interview platform designed to help candidates prepare for job interviews using intelligent resume parsing, real-time confidence analysis, sentiment detection, question generation, and performance feedback. It simulates real-world interview scenarios to improve communication skills, domain knowledge, and emotional intelligence.

## Key Features

* **AI-Powered Question Generation** based on job roles and resume content
* **Resume Parsing & Feature Extraction** using NLP models
* **Voice-Based Answer Recording** with speech-to-text conversion
* **Emotion & Confidence Analysis** using facial expression recognition and speech modulation
* **Dynamic Scoring Algorithm** for performance evaluation
* **Personalized Feedback & Recommendations**

## System Architecture

1. **User Login & Authentication**
2. **Resume Upload and Parsing Module**
3. **Dynamic Question Generation Engine**
4. **Interview Simulation Interface (Audio/Video)**
5. **Analysis Module (Emotion, Confidence, NLP Matching)**
6. **Result Evaluation & Report Generation**

## Technologies Used

* **Frontend:** HTML, CSS, JavaScript, React (optional)
* **Backend:** Python (Flask/Django)
* **AI Models:** CNN, DNN, Transformers (BERT/DistilBERT)
* **Database:** PostgreSQL / MySQL
* **APIs:** Speech Recognition API, TTS API, Emotion Detection API

## Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/MockMate.git
cd MockMate

# Create virtual environment
python -m venv env
source env/bin/activate  # On Windows use env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the development server
python app.py
```

## How It Works

1. **Upload Resume** – System extracts key skills, experience, and job role.
2. **Select Role** – User selects a job profile for the mock interview.
3. **Start Interview** – AI asks questions via voice and monitors responses.
4. **Analyze Response** – System evaluates based on correctness, tone, and facial expression.
5. **Generate Report** – Provides confidence score, improvement suggestions, and learning resources.

## Modules

### 1. Resume Parser

* NLP-based keyword extraction
* Maps candidate profile to relevant job roles

### 2. Question Generator

* Uses GPT-based models
* Adapts difficulty level in real-time

### 3. Emotion & Confidence Analyzer

* CNN for facial emotion detection
* Speech modulation to assess confidence

### 4. Feedback Engine

* Generates personalized suggestions
* Provides study resources

## Use Cases

* Job seekers preparing for interviews
* Colleges conducting placement training
* Companies assessing candidate readiness

## Future Enhancements

* Integration with LinkedIn job APIs
* Real-time mock interviewer avatar using 3D animation
* Multi-language support

