# 🎯 MockMate: AI-Based Mock Interview Preparation System

## 📌 Overview

MockMate is an AI-driven mock interview preparation platform designed to simulate real-world interview environments and provide intelligent, personalized feedback to candidates.

In today’s competitive job market, candidates often struggle with communication, confidence, and role-specific preparation. Traditional methods lack personalization and real-time evaluation. MockMate addresses these challenges using Artificial Intelligence, Natural Language Processing (NLP), and Deep Learning techniques to deliver an adaptive and realistic interview experience.

The system analyzes resumes, generates role-specific questions, evaluates verbal and non-verbal responses, and provides actionable feedback to improve candidate performance.

---

## 🚀 Key Features

- AI-driven interview question generation based on resume and job role  
- NLP-based resume parsing and skill extraction  
- Multimodal interview analysis (text, voice, and facial expressions)  
- Facial emotion detection using CNN  
- Speech confidence evaluation using RNN  
- ATS-based resume scoring and optimization suggestions  
- AI-powered aptitude test with proctoring system  
- Dynamic performance scoring and detailed feedback reports  
- Personalized improvement recommendations  
- Job role prediction and career guidance  

---

## 🧠 System Architecture

MockMate follows a modular architecture consisting of:

1. User Authentication & Profile Management  
2. Resume Upload & NLP-Based Parsing Module  
3. ATS Evaluation Engine  
4. Dynamic Question Generation Engine  
5. Interview Simulation Interface (Audio/Video)  
6. Multimodal Analysis Module  
   - Facial Emotion Detection (CNN)  
   - Speech Confidence Analysis (RNN)  
   - NLP-Based Answer Evaluation  
7. Aptitude Test Module with AI Proctoring  
   - Face detection  
   - Multiple person detection  
   - Head movement tracking  
8. Performance Evaluation & Report Generation  
9. Feedback & Recommendation Engine  

---

## 🛠 Technologies Used

### Frontend
- HTML  
- CSS  
- JavaScript  
- React (optional)  

### Backend
- Python (Flask / Django)  

### AI & Machine Learning
- CNN – Facial Emotion Recognition  
- RNN (LSTM) – Speech Confidence Analysis  
- Transformers (BERT / DistilBERT) – Resume Parsing & Answer Evaluation  

### Database
- MySQL / PostgreSQL  

### APIs & Libraries
- Speech Recognition (Google Speech / Whisper)  
- Text-to-Speech (TTS)  
- OpenCV  
- MediaPipe  
- Hugging Face Transformers  

---

## 📂 Datasets Used

### Resume Parsing & Skill Extraction
- Resume Dataset (Kaggle)  
- Indeed Job Descriptions Dataset  
- O*NET / ESCO Skills Dataset  

### Interview Question Generation
- Kaggle Interview Questions Dataset  
- Glassdoor Dataset  
- Custom Question Bank  

### Emotion Detection
- FER-2013  
- CK+ Dataset  
- RAF-DB  

### Speech & Confidence Analysis
- RAVDESS Dataset  
- CREMA-D Dataset  
- LibriSpeech  

### NLP Answer Evaluation
- SQuAD Dataset  
- STS Benchmark  
- Custom Answer Corpus  

---

## ⚙️ Installation & Setup

```bash
# Clone the repository
git clone https://github.com/your-repo/MockMate.git
cd MockMate

# Create virtual environment
python -m venv env
source env/bin/activate   # Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
