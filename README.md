# 🎯 MockMate: AI-Based Mock Interview Preparation System

## 📌 Overview

MockMate is an AI-powered mock interview preparation platform designed to simulate real-world interview environments. It helps candidates enhance their communication skills, technical knowledge, emotional intelligence, and confidence through intelligent resume parsing, adaptive question generation, emotion analysis, and detailed performance feedback.

MockMate acts as a virtual interviewer, enabling users to practice interviews anytime with AI-driven evaluation and personalized improvement recommendations.

---

## 🚀 Key Features

- AI-driven interview question generation based on job role and resume content  
- Resume parsing and skill extraction using NLP techniques  
- Voice-based interview simulation with speech-to-text conversion  
- Facial emotion and speech-based confidence analysis  
- Dynamic performance scoring system  
- Personalized feedback and improvement recommendations  

---

## 🧠 System Architecture

1. User Authentication & Profile Management  
2. Resume Upload & NLP-Based Parsing Module  
3. Dynamic Question Generation Engine  
4. Interview Simulation Interface (Audio / Video)  
5. Analysis Module  
   - Facial Emotion Detection  
   - Speech Confidence Estimation  
   - NLP-Based Answer Matching  
6. Performance Evaluation & Report Generation  

---

## 🛠 Technologies Used

### Frontend
- HTML  
- CSS  
- JavaScript  
- React (optional for enhanced UI)

### Backend
- Python (Flask / Django)

### AI & Machine Learning
- CNN – Facial emotion recognition  
- DNN – Speech confidence analysis  
- Transformers (BERT / DistilBERT) – Resume parsing and answer evaluation  

### Database
- PostgreSQL / MySQL  

### APIs & Libraries
- Speech Recognition API (Google Speech / OpenAI Whisper)  
- Text-to-Speech (TTS) API  
- OpenCV  
- MediaPipe  
- HuggingFace Transformers  

---

## 📂 Datasets Used

### 1️⃣ Resume Parsing & Skill Extraction
- Resume Dataset (Kaggle)  
- Indeed Job Descriptions Dataset  
- O*NET / ESCO Open Skills Dataset  

### 2️⃣ Interview Question Generation
- Interview Questions Dataset (Kaggle)  
- Glassdoor Interview Questions Dataset  
- Custom Curated Interview Question Bank  

### 3️⃣ Emotion Detection
- FER-2013 Dataset  
- CK+ (Cohn-Kanade Expression Dataset)  
- RAF-DB (Real-world Affective Faces Database)  

### 4️⃣ Speech & Confidence Analysis
- RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)  
- CREMA-D (Crowd-Sourced Emotional Multimodal Actors Dataset)  
- LibriSpeech Dataset  

### 5️⃣ NLP Answer Evaluation
- SQuAD Dataset  
- STS Benchmark (Semantic Textual Similarity Dataset)  
- Custom Interview Answer Corpus  

---

## ⚙️ Installation & Setup

# Clone the repository
git clone https://github.com/your-repo/MockMate.git
cd MockMate

# Create virtual environment
python -m venv env
source env/bin/activate   # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py

## 🔄 How It Works

### 1. Upload Resume
- The system extracts skills, experience, and keywords using NLP models.

### 2. Select Job Role
- Resume data is mapped against role-specific requirements.

### 3. Start Mock Interview
- AI dynamically generates and asks interview questions via voice.

### 4. Analyze Responses
- Speech relevance and fluency analysis  
- Facial expression and emotional stability detection  
- Confidence evaluation based on tone, pitch, and pauses  

### 5. Generate Performance Report
- Overall performance score  
- Identified strengths and weaknesses  
- Personalized improvement suggestions  

---

## 🧩 Core Modules

### 📄 Resume Parser
- Keyword extraction  
- Skill and experience identification  
- Role relevance analysis  

### ❓ Question Generator
- Transformer-based models  
- Real-time difficulty adjustment  

### 🎭 Emotion & Confidence Analyzer
- CNN-based facial emotion recognition  
- Speech modulation analysis (pitch, speed, hesitation)  

### 📊 Feedback & Evaluation Engine
- Multi-parameter scoring system  
- Personalized improvement recommendations  

---

## 🎯 Use Cases

- Job seekers preparing for technical and HR interviews  
- College placement training and assessment  
- Corporate candidate readiness evaluation  

