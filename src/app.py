import os
import time
import json
import logging
import tempfile
from functools import wraps
from io import BytesIO
import base64
from flask import (
    Flask, render_template, request, redirect, url_for,
    session, flash, jsonify, send_from_directory
)
import mysql.connector
from werkzeug.security import generate_password_hash, check_password_hash
import torch
import torch.nn as nn
import numpy as np
import requests
import re
from PIL import Image
import cv2
from torchvision import models
import torchvision.transforms as T
try:
    import speech_recognition as sr
    HAS_SPEECHREC = True
except Exception:
    HAS_SPEECHREC = False
from ravdees_confidence import (
    ConfidenceBiLSTM,
    DEVICE as CONF_DEVICE,
    pad_or_truncate,
    extract_mfcc,
    load_audio,
    CONFIDENCE_MAP
)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("mockmate")
app = Flask(__name__)
app.secret_key = os.getenv("MM_SECRET", "dev-secret-changeme")
UPLOAD_DIR = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASS", "dhivya@2005"),
    "database": os.getenv("DB_NAME", "mockmate_db"),
    "raise_on_warnings": True,
}
def get_db_connection():
    return mysql.connector.connect(**DB_CONFIG)
def login_required_json(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if 'user' not in session:
            return jsonify({"error": "not_authenticated"}), 403
        return fn(*args, **kwargs)
    return wrapper
MODEL_PATH = os.path.join(os.getcwd(), "job_role_model")
try:
    from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
    if os.path.isdir(MODEL_PATH):
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH, local_files_only=True)
        hf_model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH, local_files_only=True)
        id2label = torch.load(os.path.join(MODEL_PATH, "id2label.pt"), map_location="cpu")
        log.info("Loaded local job_role_model from %s", MODEL_PATH)
    else:
        tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
        hf_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
        id2label = {0: "Unknown"}
        log.info("Loaded fallback DistilBERT from hub")
except Exception as e:
    log.exception("Failed to load HuggingFace model: %s", e)
    tokenizer = None
    hf_model = None
    id2label = {}
import os
import requests
import logging

log = logging.getLogger(__name__)

GOOGLE_GEMINI_API_KEY = os.getenv("GOOGLE_GEMINI_API_KEY")

GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
)

def call_gemini(prompt_text, timeout=15):
    if not GOOGLE_GEMINI_API_KEY:
        log.warning("Gemini API key not set")
        return None

    params = {
        "key": GOOGLE_GEMINI_API_KEY
    }

    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt_text}
                ]
            }
        ]
    }

    try:
        resp = requests.post(
            GEMINI_ENDPOINT,
            params=params,
            headers=headers,
            json=payload,
            timeout=timeout
        )

        resp.raise_for_status()
        data = resp.json()

        return data["candidates"][0]["content"]["parts"][0]["text"]

    except Exception as e:
        log.exception("Gemini call failed: %s", e)
        return None



def clean_question_text(q):
    return re.sub(r'^\s*\d+[\.\)\-\s]*', '', q).strip()
def parse_questions_with_links(gemini_text, expected=5):
    if not gemini_text:
        return []
    unwanted_phrases = ["could not generate question", "unable to generate question", "no question generated"]
    def is_valid_line(line):
        return not any(phrase in line.lower() for phrase in unwanted_phrases)
    try:
        parsed = json.loads(gemini_text)
        if isinstance(parsed, list):
            out = []
            for item in parsed:
                if isinstance(item, dict):
                    q = (item.get('question') or item.get('q') or item.get('text') or "").strip()
                    link = (item.get('link') or item.get('url') or "").strip()
                    if q and is_valid_line(q):
                        out.append({'question': q, 'link': link})
            return out[:expected]
    except Exception:
        pass
    lines = [ln.strip() for ln in gemini_text.splitlines() if ln.strip()]
    lines = [ln for ln in lines if is_valid_line(ln)]
    out = []
    i = 0
    while i < len(lines) and len(out) < expected:
        ln = lines[i]
        if ln.endswith('?'):
            q = clean_question_text(ln)
            link = ""
            m = re.search(r'(https?://\S+|www\.\S+)', ln)
            if m:
                link = m.group(0)
            elif i + 1 < len(lines):
                m2 = re.search(r'(https?://\S+|www\.\S+)', lines[i + 1])
                if m2:
                    link = m2.group(0)
                    i += 1
            out.append({'question': q, 'link': link})
        else:
            m = re.search(r'([A-Z].*\?)', ln)
            if m:
                q = clean_question_text(m.group(1))
                link = ""
                mm = re.search(r'(https?://\S+|www\.\S+)', ln)
                if mm:
                    link = mm.group(0)
                out.append({'question': q, 'link': link})
        i += 1
    return out[:expected]
CONF_CHECKPOINT = os.path.join(os.getcwd(), "confidence_bilstm.pth")
try:
    log.info("Loading confidence model from %s", CONF_CHECKPOINT)
    confidence_model = ConfidenceBiLSTM().to(CONF_DEVICE)
    try:
        chk = torch.load(CONF_CHECKPOINT, map_location=CONF_DEVICE, weights_only=False)
    except TypeError:
        chk = torch.load(CONF_CHECKPOINT, map_location=CONF_DEVICE)
    if isinstance(chk, dict) and 'model_state_dict' in chk:
        state = chk['model_state_dict']
    elif isinstance(chk, dict) and all(k.startswith('lstm') or k.startswith('classifier') or k.startswith('regressor') or '.' in k for k in chk.keys()):
        # heuristics: looks like a state_dict
        state = chk
    else:
        state = chk
    confidence_model.load_state_dict(state)
    confidence_model.eval()
    log.info("Confidence model loaded successfully.")
except Exception as e:
    log.exception("Failed to load confidence model: %s", e)
    confidence_model = None
EMOTION_MODEL_PATH = os.path.join(os.getcwd(), "affectnet_resnet50_amp.pth")
HAAR_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(HAAR_PATH)

emotion_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])
try:
    EMOTION_CLASSES = torch.load(os.path.join(os.getcwd(), "emotion_classes.pt"))
    if not isinstance(EMOTION_CLASSES, (list, tuple)):
        raise ValueError("emotion_classes.pt does not contain a list")
except Exception:
    EMOTION_CLASSES = ["neutral", "happy", "sad", "surprise", "fear", "disgust", "anger"]
emotion_model = None
EM_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try:
    log.info("Loading emotion model from %s", EMOTION_MODEL_PATH)
    em_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_ftrs = em_model.fc.in_features
    em_model.fc = nn.Linear(num_ftrs, len(EMOTION_CLASSES))  # 7 classes
    chk = torch.load(EMOTION_MODEL_PATH, map_location="cpu")
    if isinstance(chk, dict) and 'model_state_dict' in chk:
        state = chk['model_state_dict']
    else:
        state = chk
    loaded_state = {}
    for k, v in state.items():
        if k.startswith('fc.') and v.shape != em_model.state_dict()[k].shape:
            log.warning("Skipping fc layer weights from checkpoint due to size mismatch: %s", k)
            continue
        loaded_state[k] = v
    em_model.load_state_dict(loaded_state, strict=False)  # load rest of weights
    em_model.eval()
    em_model.to(EM_DEVICE)
    emotion_model = em_model
    log.info("Emotion model loaded on %s", EM_DEVICE)
except Exception as e:
    log.exception("Failed to load emotion model: %s", e)
    emotion_model = None
def decode_base64_image(data_url):
    try:
        header, b64 = data_url.split(',', 1)
    except Exception:
        b64 = data_url
    img_bytes = base64.b64decode(b64)
    return Image.open(BytesIO(img_bytes)).convert("RGB")
def detect_and_crop_face_pil(pil_img):
    # Convert to numpy BGR for OpenCV
    np_img = np.array(pil_img)[:, :, ::-1].copy()  # RGB -> BGR
    gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50,50))
    if len(faces) == 0:
        return None
    faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
    x, y, w, h = faces[0]
    pad = int(0.2 * max(w, h))
    x1 = max(0, x - pad); y1 = max(0, y - pad)
    x2 = min(np_img.shape[1], x + w + pad); y2 = min(np_img.shape[0], y + h + pad)
    face_bgr = np_img[y1:y2, x1:x2]
    face_rgb = face_bgr[:, :, ::-1]
    return Image.fromarray(face_rgb)
def predict_emotion_from_pil(face_pil):
    if emotion_model is None:
        return None
    try:
        inp = emotion_transform(face_pil).unsqueeze(0).to(EM_DEVICE)
        with torch.no_grad():
            logits = emotion_model(inp)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            idx = int(np.argmax(probs))
            label = EMOTION_CLASSES[idx] if 0 <= idx < len(EMOTION_CLASSES) else str(idx)
            score = float(probs[idx])
        return {"label": label, "score": round(score, 4), "probs": probs.tolist()}
    except Exception as e:
        log.exception("Emotion prediction failed: %s", e)
        return None
@app.route('/')
def home():
    return redirect(url_for('login'))
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        if not (name and email and password):
            flash("All fields required", "danger")
            return redirect(url_for('register'))
        hashed = generate_password_hash(password)
        db = get_db_connection()
        cursor = db.cursor()
        cursor.execute("SELECT id FROM users WHERE email=%s", (email,))
        if cursor.fetchone():
            flash("Email already registered", "danger")
            cursor.close(); db.close(); return redirect(url_for('register'))
        cursor.execute("INSERT INTO users (name,email,password_hash) VALUES (%s,%s,%s)", (name,email,hashed))
        db.commit()
        cursor.close(); db.close()
        flash("Registered. Please login.", "success")
        return redirect(url_for('login'))
    return render_template('register.html')
@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email','').strip()
        password = request.form.get('password','')
        db = get_db_connection()
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE email=%s", (email,))
        user = cursor.fetchone()
        cursor.close(); db.close()
        if user and check_password_hash(user['password_hash'], password):
            session['user'] = user['name']
            session['user_id'] = user['id']
            flash("Login successful", "success")
            return redirect(url_for('upload_resume'))
        flash("Invalid credentials", "danger")
        return redirect(url_for('login'))
    return render_template('login.html')
@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out", "success")
    return redirect(url_for('login'))
@app.route('/upload', methods=['GET','POST'])
def upload_resume():
    if 'user' not in session:
        flash("Please log in", "danger")
        return redirect(url_for('login'))
    if request.method == 'POST':
        if 'resume' not in request.files:
            flash("No file uploaded", "danger")
            return redirect(url_for('upload_resume'))
        file = request.files['resume']
        if file.filename == '':
            flash("No file selected", "danger")
            return redirect(url_for('upload_resume'))
        if not file.filename.lower().endswith('.pdf'):
            flash("Only PDF allowed", "danger"); return redirect(url_for('upload_resume'))
        fn = f"{int(time.time()*1000)}_{file.filename}"
        path = os.path.join(UPLOAD_DIR, fn)
        file.save(path)
        try:
            from pdfminer.high_level import extract_text
            resume_text = extract_text(path)
        except Exception as e:
            log.exception("PDF extraction failed: %s", e)
            resume_text = ""
        candidate_summary = resume_text[:1000].replace('\n',' ')
        # predict role using HF model (if available)
        top_role_name = "Unknown Role"
        top_prob = 0.0
        if tokenizer and hf_model:
            try:
                inputs = tokenizer(resume_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
                outputs = hf_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)[0]
                topk = torch.topk(probs, k=min(4, probs.shape[0]))
                top_indices = topk.indices.tolist()
                top_probs = [round(probs[i].item()*100, 2) for i in top_indices]
                top_roles = [id2label.get(i, f"role_{i}") for i in top_indices]
                session['predicted_roles'] = [{'role':r,'prob':p} for r,p in zip(top_roles, top_probs)]
                top_role_name = top_roles[0]
                top_prob = top_probs[0]
            except Exception as e:
                log.exception("Role prediction failed: %s", e)
        else:
            session['predicted_roles'] = []
        reason = "Reason unavailable"
        try:
            reason = (call_gemini(f"The candidate has these skills and experience: {candidate_summary}."
                                   f" Predicted job role: {top_role_name}. In ~150 words describe why this candidate fits." , timeout=10)
                      or "Reason explanation is currently unavailable.")
        except Exception:
            pass
        session['chat_history'] = []
        return render_template('result.html', role=top_role_name, confidence=top_prob, reason=reason, filename=fn)
    return render_template('upload.html')
@app.route('/questions')
def questions():
    if 'predicted_roles' not in session:
        flash("No predicted role found. Please upload a resume first.", "danger")
        return redirect(url_for('upload_resume'))
    top_role = session['predicted_roles'][0]['role'] if session.get('predicted_roles') else "Candidate"
    intro_text = call_gemini(f"Write a short intro welcoming a candidate to an interview for the role {top_role}.", timeout=6) or ""
    return render_template('questions.html', role=top_role, intro=intro_text)
@app.route('/get_questions', methods=['POST'])
@login_required_json
def get_questions():
    data = request.get_json() or {}
    category = data.get('category')
    override_role = data.get('role')
    if category == 'job_role':
        roles = session.get('predicted_roles', [])
        if not roles:
            return jsonify({"error": "no_predicted_role"}), 400
        role_name = roles[0]['role']
        prompt = (f"Generate exactly 5 short, clear interview questions for the job role: {role_name}. "
                  "For each question, also provide one concise, trustworthy online resource link (URL). "
                  "Return a JSON array like [{\"question\":\"...\",\"link\":\"https://...\"}, ...].")
        label = f"Job Role Questions — {role_name}"
    elif category == 'hr':
        prompt = ("Generate exactly 5 common HR / behavioral interview questions. "
                  "For each question include a short resource link (URL). "
                  "Return JSON array of objects.")
        label = "HR Questions"
    elif category == 'other':
        roles = session.get('predicted_roles', [])
        other_roles = [r for r in roles[1:]] if len(roles) > 1 else []
        if override_role:
            prompt = (f"Generate exactly 5 interview questions for the job role: {override_role}. "
                      "Also provide a short resource link for each question. Return JSON array.")
            label = f"Other Role Questions — {override_role}"
        else:
            return jsonify({"other_roles": other_roles})
    else:
        return jsonify({"error": "invalid_category"}), 400
    gemini_text = call_gemini(prompt, timeout=15)
    questions = parse_questions_with_links(gemini_text, expected=5)
    chat = session.get('chat_history', [])
    system_entry = {"timestamp": int(time.time()), "source": "system", "label": label, "questions": questions}
    chat.append(system_entry)
    session['chat_history'] = chat
    return jsonify({"questions": questions, "label": label})
@app.route('/track_emotion', methods=['POST'])
@login_required_json
def track_emotion():
    """
    Expects JSON: { "frame": "data:image/jpeg;base64,..." , "ts": 1234567890 }
    Returns: { "emotion": "happy", "score": 0.87 }
    """
    if emotion_model is None:
        return jsonify({"error": "emotion_model_not_loaded"}), 500

    data = request.get_json() or {}
    frame = data.get('frame')
    ts = int(data.get('ts', time.time()*1000))
    if not frame:
        return jsonify({"error": "no_frame"}), 400

    try:
        pil = decode_base64_image(frame)
        face = detect_and_crop_face_pil(pil)
        if face is None:
            resp = {"emotion": "no_face", "score": 0.0}
        else:
            pred = predict_emotion_from_pil(face)
            if pred is None:
                resp = {"emotion": "error", "score": 0.0}
            else:
                resp = {"emotion": pred["label"], "score": pred["score"]}
        # save to session emotion stream
        stream = session.get('emotion_stream', [])
        stream.append({"ts": ts, "emotion": resp["emotion"], "score": resp["score"]})
        # keep only last N to avoid bloat
        session['emotion_stream'] = stream[-500:]
        return jsonify(resp)
    except Exception as e:
        log.exception("Emotion tracking failed: %s", e)
        return jsonify({"error": "processing_failed", "detail": str(e)}), 500
@app.route('/evaluate_answer', methods=['POST'])
@login_required_json
def evaluate_answer():
    if 'audio' not in request.files:
        return jsonify({"error": "no_audio"}), 400
    file = request.files['audio']
    if file.filename == '':
        return jsonify({"error": "empty_file"}), 400
    ts = int(time.time()*1000)
    tmp_name = f"resp_{ts}.wav"
    tmp_path = os.path.join(UPLOAD_DIR, tmp_name)
    file.save(tmp_path)
    if confidence_model is None:
        log.error("Confidence model not available")
        return jsonify({"error": "model_not_loaded"}), 500
    try:
        y = load_audio(tmp_path)
        mfcc = extract_mfcc(y)
        mfcc = pad_or_truncate(mfcc)
        mfcc = (mfcc - np.mean(mfcc, axis=0, keepdims=True)) / (np.std(mfcc, axis=0, keepdims=True) + 1e-9)
        x = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).to(next(confidence_model.parameters()).device)
        with torch.no_grad():
            logits, conf = confidence_model(x)
            pred_conf = float(conf.item())
            pred_class = int(torch.argmax(logits, dim=1).item())
            emotion_names = list(CONFIDENCE_MAP.keys())
            conf_emotion_label = emotion_names[pred_class] if 0 <= pred_class < len(emotion_names) else "unknown"
    except Exception as e:
        log.exception("Confidence inference failed: %s", e)
        return jsonify({"error": "inference_error", "detail": str(e)}), 500
    transcribed = ""
    if HAS_SPEECHREC:
        try:
            r = sr.Recognizer()
            with sr.AudioFile(tmp_path) as source:
                aud = r.record(source)
                transcribed = r.recognize_google(aud)
        except Exception as e:
            log.info("Transcription failed: %s", e)
    latest_em = {"emotion": "unknown", "emotion_score": 0.0}
    try:
        em_stream = session.get('emotion_stream', [])
        if em_stream:
            last = em_stream[-1]
            latest_em = {"emotion": last.get('emotion', 'unknown'), "emotion_score": float(last.get('score', 0.0))}
    except Exception:
        pass
    saved = session.get('answers', [])
    current_answer = {
        "ts": ts,
        "file": tmp_name,
        "transcription": transcribed,
        "confidence": round(pred_conf, 2),
        "confidence_emotion_label": conf_emotion_label,
        "emotion": latest_em["emotion"],
        "emotion_score": round(latest_em["emotion_score"], 4)
    }
    saved.append(current_answer)
    session['answers'] = saved
    return jsonify(current_answer)
@app.route('/chat_history', methods=['GET'])
@login_required_json
def chat_history():
    return jsonify({"chat_history": session.get('chat_history', []), "answers": session.get('answers', []), "emotion_stream": session.get('emotion_stream', [])})
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_DIR, filename, as_attachment=False)
@app.route('/aptitude', methods=['GET','POST'])
def aptitude_category():
    if 'user' not in session:
        flash("Please log in first.", "danger"); return redirect(url_for('login'))
    categories = ['Quantitative','Verbal','Logical']
    if request.method == 'POST':
        selected_category = request.form.get('category')
        if selected_category not in categories:
            flash("Please select a valid category.", "danger"); return redirect(url_for('aptitude_category'))
        return redirect(url_for('aptitude_test', category=selected_category))
    return render_template('aptitude_category.html', categories=categories)
@app.route('/aptitude/test/<category>', methods=['GET','POST'])
def aptitude_test(category):
    if 'user' not in session:
        flash("Please log in first.", "danger"); return redirect(url_for('login'))
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    if request.method == 'POST':
        submitted_answers = request.form.to_dict()
        results = []; correct_count = 0
        for qid_str, user_ans in submitted_answers.items():
            try:
                qid = int(qid_str)
            except:
                continue
            cursor.execute("SELECT * FROM aptitude_questions WHERE id=%s", (qid,))
            q = cursor.fetchone()
            if not q:
                continue
            is_correct = (user_ans == q['correct_answer'])
            if is_correct:
                correct_count += 1
            results.append({
                "question": q['question'],
                "options": {"A": q['option_a'], "B": q['option_b'], "C": q['option_c'], "D": q['option_d']},
                "selected": user_ans,
                "correct": q['correct_answer'],
                "is_correct": is_correct
            })
        total = len(results)
        score = round((correct_count / total) * 100, 2) if total > 0 else 0
        cursor.close(); db.close()
        return render_template('aptitude_result.html', results=results, total=total, correct=correct_count, score=score)
    cursor.execute("SELECT * FROM aptitude_questions WHERE category=%s ORDER BY RAND() LIMIT 10", (category,))
    questions = cursor.fetchall()
    cursor.close(); db.close()
    return render_template('aptitude_test.html', category=category, questions=questions)
import json
def fetch_ai_learning_resources(role_name, avg_confidence):
    role_name = role_name.lower().strip()
    role_courses = {
        "data science": [
            {"title": "Python for Data Science", "url": "https://www.coursera.org/learn/python-data-analysis?utm_medium=institutions&utm_source=umich&utm_campaign=adwords-india-applied-data-science-with-python-intro-to-data-science-with-python&utm_term=how%20to%20use%20python%20in%20data%20science&gad_source=1&gad_campaignid=1055523834&gbraid=0AAAAADR2vnrziLeKiVUrVJ_2EulArxbl9&gclid=CjwKCAjwup3HBhAAEiwA7euZuts4_YTYKQpDntYT8xKs30_xA-zLTttRAOJ_VGrVwt6FZ7OHAnNdQBoCWaYQAvD_BwE", "reason": "Master Python for data analysis and ML."},
            {"title": "Machine Learning A-Z", "url": "https://www.udemy.com/course/machinelearning/?utm_source=adwords&utm_medium=udemyads&utm_campaign=Search_DSA_Alpha_Prof_la.EN_cc.India_Subs&campaigntype=Search&portfolio=India&language=EN&product=Subs&test=&audience=DSA&topic=Data_Science&priority=Alpha&utm_content=deal4584&utm_term=_._ag_185390583513_._ad_769665429056_._kw__._de_c_._dm__._pl__._ti_dsa-2436670444979_._li_9148687_._pd__._&matchtype=&gad_source=1&gad_campaignid=22900574864&gbraid=0AAAAADROdO3x9SlPFkZIYv6EHVzaLQEl2&gclid=CjwKCAjwup3HBhAAEiwA7euZuv5I3xIh2sJr2oH6I0yzZDmVKXBtbRKX48MqDZm5pJmnWBtjxrYl7RoCCAwQAvD_BwE&couponCode=PMNVD2025", "reason": "Comprehensive guide to machine learning."},
            {"title": "SQL for Data Analysis", "url": "https://www.udemy.com/course/mysql-for-data-analysis/?utm_source=adwords&utm_medium=udemyads&utm_campaign=Search_Keyword_Alpha_Prof_la.DE_cc.ROW-German&campaigntype=Search&portfolio=ROW-German&language=DE&product=Course&test=&audience=Keyword&topic=SQL&priority=Alpha&utm_content=deal4584&utm_term=_._ag_173526752788_._ad_706339838828_._kw_sql+for+data+science_._de_c_._dm__._pl__._ti_kwd-302992780302_._li_9148687_._pd__._&matchtype=b&gad_source=1&gad_campaignid=21479364228&gbraid=0AAAAADROdO2HNBvCcxiypQTn92MgvmElS&gclid=CjwKCAjwup3HBhAAEiwA7euZuiJVxfXR5PF7kdzPXfiCOGHg-lOAZ8jcYtlE0i924Nhz2EjPniJBpBoC5SsQAvD_BwE&couponCode=PMNVD2025", "reason": "Learn SQL to query and analyze data."},
            {"title": "Statistics for Data Science", "url": "https://www.udemy.com/course/the-data-science-course-complete-data-science-bootcamp/?utm_source=adwords&utm_medium=udemyads&utm_campaign=Search_Keyword_Alpha_Prof_la.EN_cc.India&campaigntype=Search&portfolio=India&language=EN&product=Course&test=&audience=Keyword&topic=Data_Science&priority=Alpha&utm_content=deal4584&utm_term=_._ag_160270533745_._ad_713313366149_._kw_statistics+for+data+science_._de_c_._dm__._pl__._ti_kwd-68690028279_._li_9148687_._pd__._&matchtype=b&gad_source=1&gad_campaignid=21178559968&gbraid=0AAAAADROdO3wmJnVQT_H9RIzrL7-nLIlV&gclid=CjwKCAjwup3HBhAAEiwA7euZuqjs3f7O9E1_pYxL6Pno0ylJ_5L8UzkXEIZFIZ3B_uZDqGp8qaiY6BoCydwQAvD_BwE&couponCode=PMNVD2025", "reason": "Build a solid math foundation for AI."},
            {"title": "Deep Learning Specialization", "url": "https://www.coursera.org/specializations/deep-learning?&utm_medium=sem&utm_source=gg&utm_campaign=b2c_india_deep-learning_deeplearning-ai_ftcof_specializations_cx_dr_bau_gg_pmax_pr_in_all_m_hyb_24-01_desktop&campaignid=20920191021&adgroupid=&device=c&keyword=&matchtype=&network=x&devicemodel=&creativeid=&assetgroupid=6487004747&targetid=&extensionid=&placement=&gad_source=1&gad_campaignid=20930155807&gbraid=0AAAAADdKX6bVVjHbSEo6iatlDfAJJ2DYt&gclid=CjwKCAjwup3HBhAAEiwA7euZunqBYCcKKMZksc6qRmq0IH61Xf68fRJGtxRVF7KwXPNLwWDoK6xPdRoCb38QAvD_BwE", "reason": "Develop neural network-based AI models."}
        ],
        "hr": [
            {"title": "Human Resource Management", "url": "https://www.coursera.org/articles/human-resources?utm_medium=sem&utm_source=gg&utm_campaign=b2c_india_google-it-automation_google_ftcof_professional-certificates_cx_dr_bau_gg_pmax_pr_in_all_m_hyb_22-11_desktop&campaignid=19197733182&adgroupid=&device=c&keyword=&matchtype=&network=x&devicemodel=&creativeid=&assetgroupid=6458849661&targetid=&extensionid=&placement=&gad_source=1&gad_campaignid=19204410364&gbraid=0AAAAADdKX6av6S_WBTtgls7sWOPHCKzEQ&gclid=CjwKCAjwup3HBhAAEiwA7euZuqIS_V9NE2gHZiZLlgrSqm14rQIOFoZOLSeEcW0KxvpOEgq2S0bvphoCwrgQAvD_BwE", "reason": "Understand HR strategies and frameworks."},
            {"title": "HR Analytics using Excel", "url": "https://www.udemy.com/course/hr-analytics-using-excel/?source=mrtechnawy.com&amp%3BcouponCode=HRAFRSP1&im_ref=x380%3AaWl6xycWPoRPr1DozJvUkp3553VqW9SW00&sharedid=118678&irpid=1453307&utm_medium=affiliate&utm_source=impact&utm_audience=mx&utm_tactic=&utm_content=3281534&utm_campaign=1453307&irgwc=1&gad_source=1", "reason": "Leverage data for HR insights."},
            {"title": "People Analytics", "url": "https://www.coursera.org/learn/wharton-people-analytics?utm_medium=sem&utm_source=gg&utm_campaign=b2c_india_google-it-automation_google_ftcof_professional-certificates_cx_dr_bau_gg_pmax_pr_in_all_m_hyb_22-11_desktop&campaignid=19197733182&adgroupid=&device=c&keyword=&matchtype=&network=x&devicemodel=&creativeid=&assetgroupid=6458849661&targetid=&extensionid=&placement=&gad_source=1&gad_campaignid=19204410364&gbraid=0AAAAADdKX6av6S_WBTtgls7sWOPHCKzEQ&gclid=CjwKCAjwup3HBhAAEiwA7euZuguauJmfZX5M2g2SaKU9bDibHLn_jZSSmkUC_Wxf-mjYO--BXfVOjxoChHgQAvD_BwE", "reason": "Learn data-driven HR decision-making."},
            {"title": "Communication in Workplace", "url": "https://www.coursera.org/in/articles/workplace-communication", "reason": "Enhance workplace communication."},
            {"title": "Talent Management", "url": "https://www.coursera.org/in/articles/talent-management", "reason": "Develop strategies for employee growth."}
        ],
        "advocate": [
            {"title": "Legal Research and Writing", "url": "https://www.manupatracademy.com/home/legal-research-and-writingh", "reason": "Improve legal documentation and research skills."},
            {"title": "Cyber Law and Privacy", "url": "https://www.udemy.com/course/cyber-crime-cyber-law-by-dr-pavan-duggal/?srsltid=AfmBOopP5SAjawKC1eW6HIOlPvLYX9K6iIn6k7qTVKH-JvTCE5xmAawe&couponCode=25BBPMXNVD35", "reason": "Understand data protection and cybercrime laws."},
            {"title": "Intellectual Property Law", "url": "https://www.law.georgetown.edu/your-life-career/career-exploration-professional-development/for-jd-students/explore-legal-careers/practice-areas/intellectual-property-law/", "reason": "Explore IP law essentials."},
            {"title": "Contract Law Fundamentals", "url": "https://www.udemy.com/course/contract-law/?srsltid=AfmBOopiyXryUXn883Ey28C1UszbrHcr5v9UpxcmLu-18yyMnqliWmky&couponCode=25BBPMXNVD35", "reason": "Learn how contracts are formed and enforced."},
            {"title": "International Business Law", "url": "https://www.udemy.com/topic/business-law/?srsltid=AfmBOooDZqa5lBW_N2iM991QM2yBxa38AjHFeBGi4HSZoyawz1a2DRdr", "reason": "Understand legal aspects of global trade."}
        ],
        "arts": [
            {"title": "Modern Art & Ideas", "url": "https://www.coursera.org/learn/modern-art-ideas", "reason": "Explore modern art movements and history."},
            {"title": "Creative Thinking", "url": "https://in.indeed.com/career-advice/career-development/creative-thinking", "reason": "Develop creative skills for artistic expression."},
            {"title": "Digital Art for Beginners", "url": "https://www.udemy.com/course/digital-art-for-beginners-unleash-your-creativity/?srsltid=AfmBOoq6zvwO67CFjhbg6cR0B4ThlmDxoTgN4kplMWlP6zZPhdF9rsup&couponCode=25BBPMXNVD35", "reason": "Learn digital painting and drawing basics."},
            {"title": "Graphic Design Basics", "url": "https://www.coursera.org/learn/fundamentals-of-graphic-design", "reason": "Master the principles of design."},
            {"title": "History of Art", "url": "https://en.wikipedia.org/wiki/History_of_art", "reason": "Understand evolution of art through eras."}
        ],
        "web designing": [
            {"title": "Responsive Web Design", "url": "https://www.freecodecamp.org/learn/responsive-web-design/", "reason": "Build responsive websites with HTML & CSS."},
            {"title": "Web Design for Beginners", "url": "https://www.geeksforgeeks.org/websites-apps/getting-started-with-web-design/", "reason": "Learn web design fundamentals."},
            {"title": "Advanced CSS and Sass", "url": "https://www.udemy.com/course/advanced-css-and-sass/?utm_source=adwords&utm_medium=udemyads&utm_campaign=Search_DSA_Beta_Prof_la.EN_cc.India_Subs&campaigntype=Search&portfolio=India&language=EN&product=Subs&test=&audience=DSA&topic=&priority=Beta&utm_content=deal4584&utm_term=_._ag_185390585033_._ad_769665429293_._kw__._de_c_._dm__._pl__._ti_dsa-2436670172859_._li_9148661_._pd__._&matchtype=&gad_source=1&gad_campaignid=22900574867&gbraid=0AAAAADROdO0xQLUr4wUDaniNgP-kJwZ-r&gclid=CjwKCAjwup3HBhAAEiwA7euZujlMlx_Ialtlg2TDFjlBjh5ZhoWY4LZdu6B_NMrjZbnApHzxi3AZeRoC9GgQAvD_BwE&couponCode=PMNVD2025", "reason": "Enhance front-end designs with animations."},
            {"title": "Figma UI/UX Design", "url": "https://www.udemy.com/course/figma-ui-ux-design-advanced-tutorial/?utm_source=adwords&utm_medium=udemyads&utm_campaign=Search_DSA_Beta_Prof_la.EN_cc.India_Subs&campaigntype=Search&portfolio=India&language=EN&product=Subs&test=&audience=DSA&topic=&priority=Beta&utm_content=deal4584&utm_term=_._ag_185390584753_._ad_769665429269_._kw__._de_c_._dm__._pl__._ti_dsa-2436670174299_._li_9148661_._pd__._&matchtype=&gad_source=1&gad_campaignid=22900574867&gbraid=0AAAAADROdO0xQLUr4wUDaniNgP-kJwZ-r&gclid=CjwKCAjwup3HBhAAEiwA7euZumTGtDK2Rab2PvVOFiwcTnkw_5i8InMLZqY1kuuyz4kO6z5v7DBt3hoCcc0QAvD_BwE&couponCode=PMNVD2025", "reason": "Design professional UI layouts."},
            {"title": "React Frontend Development", "url": "https://www.coursera.org/learn/frontend-development-using-react", "reason": "Build interactive front-end apps."}
        ],
        "mechanical engineer": [
            {"title": "MATLAB for Engineers", "url": "https://matlabacademy.mathworks.com/?page=1&sort=featured", "reason": "Analyze engineering problems using MATLAB."},
            {"title": "AutoCAD Mechanical Design", "url": "https://www.udemy.com/course/autocad-2021-mechanical-2d-and-3d-for-beginners-to-advanced/?srsltid=AfmBOoqT-cKp3rp3OqvkBtvrv4m8NxBFiKajl3k4ab7N-ichKfEKJ6g6&couponCode=25BBPMXNVD35", "reason": "Learn CAD modeling and simulation."},
            {"title": "Thermodynamics", "url": "https://www.coursera.org/learn/thermodynamics-intro", "reason": "Understand heat and energy systems."},
            {"title": "SolidWorks 3D Modeling", "url": "https://www.coursera.org/specializations/practice-solidworks-3d-cad", "reason": "Design mechanical parts with SolidWorks."},
            {"title": "Robotics Foundations", "url": "https://www.coursera.org/courses?query=robotics", "reason": "Explore automation and robotics concepts."}
        ],
        "sales": [
            {"title": "Sales Training Masterclass", "url": "https://www.udemy.com/topic/sales-skills/?srsltid=AfmBOor0LrmxNatqa8ukL3VnneNivgLcB42obmjL1FdwJA5CqNspxAEo", "reason": "Learn professional sales techniques."},
            {"title": "Negotiation Skills", "url": "https://www.coursera.org/learn/negotiation-skills", "reason": "Master negotiation and persuasion."},
            {"title": "CRM with Salesforce", "url": "https://www.salesforce.com/crm/free-trial/", "reason": "Learn CRM operations using Salesforce."},
            {"title": "Marketing and Consumer Behavior", "url": "https://www.coursera.org/courses?query=consumer%20behavior", "reason": "Understand buyer psychology."},
            {"title": "Business Communication", "url": "https://www.coursera.org/courses?query=business%20communication", "reason": "Develop effective communication for sales."}
        ],
        "health and fitness": [
            {"title": "Nutrition and Lifestyle", "url": "https://www.coursera.org/browse/health/nutrition", "reason": "Learn diet and wellness management."},
            {"title": "Personal Trainer Certification", "url": "https://www.acefitness.org/fitness-certifications/personal-trainer-certification/b/?srsltid=AfmBOoqR2ep5miIk7psLsYM9yz2lljbNYOfrPutJdKiL9KW8Xj_WpsIV&mrasn=1508475.1879295.E47Q9Jif", "reason": "Prepare for a fitness coaching career."},
            {"title": "Yoga and Mindfulness", "url": "https://sivananda.org.in/?gad_source=1&gad_campaignid=22058169722&gbraid=0AAAAADjbloAyHgXPhgfMwoA_5JkNFVtCX&gclid=CjwKCAjwup3HBhAAEiwA7euZuovlmLh_yCT3CzLoUPOgHX2o4jkybSx8ReOio7ULPMtyICMf3pKOrhoCEREQAvD_BwE", "reason": "Practice mindfulness and yoga techniques."},
            {"title": "Exercise Physiology", "url": "https://en.wikipedia.org/wiki/Exercise_physiology", "reason": "Understand how the human body adapts to exercise."},
            {"title": "Mental Health Awareness", "url": "https://www.who.int/health-topics/mental-health#tab=tab_1", "reason": "Promote psychological well-being."}
        ],
        "civil engineer": [
            {"title": "Construction Management", "url": "https://www.coursera.org/in/articles/construction-management", "reason": "Master project management in civil works."},
            {"title": "Revit Architecture", "url": "https://www.udemy.com/course/revit-architecture-an-ultimate-guide/?srsltid=AfmBOoolBXXd7pwRvX3mReB7F74o_JbvWlHZp9q4y7qYkqWDuyRg5esF&couponCode=PMNVD2025", "reason": "Learn 3D BIM modeling."},
            {"title": "Structural Analysis", "url": "https://en.wikipedia.org/wiki/Structural_analysis", "reason": "Analyze load and stress in structures."},
            {"title": "AutoCAD for Civil Engineers", "url": "https://www.udemy.com/course/autocad-for-civil-engineers/?srsltid=AfmBOoo0lVLomFIHfW6kqQn7KBVxjFY-2dBJAwFBBkFtY3hQxsvGhbG3&couponCode=PMNVD2025", "reason": "Draw and model civil designs."},
            {"title": "Geotechnical Engineering", "url": "https://en.wikipedia.org/wiki/Geotechnical_engineering", "reason": "Understand soil and foundation mechanics."}
        ],
        "java developer": [
            {"title": "Java Programming Masterclass", "url": "https://www.udemy.com/course/java-the-complete-java-developer-course/?srsltid=AfmBOoq7c1PfTUrh2l-fSohd6wdndOUkkybB5ZbbR0tXI5bllkw8hQbX", "reason": "Master Java programming end-to-end."},
            {"title": "Spring Boot Microservices", "url": "https://www.coursera.org/learn/google-cloud-java-spring", "reason": "Build scalable web apps in Java."},
            {"title": "Data Structures & Algorithms in Java", "url": "https://www.coursera.org/courses?query=java%20data%20structures", "reason": "Strengthen your DSA skills."},
            {"title": "RESTful API with Java", "url": "httaps://www.oracle.com/technical-resources/articles/java/jax-rs.html", "reason": "Develop RESTful APIs using Spring Boot."},
            {"title": "Advanced Java", "url": "https://www.coursera.org/learn/advanced-java-certification-course", "reason": "Learn advanced topics in Java development."}
        ],
        "business analyst": [
            {"title": "Business Analysis Fundamentals", "url": "https://www.udemy.com/course/the-business-intelligence-analyst-course-2018/?utm_source=adwords&utm_medium=udemyads&utm_campaign=Search_Keyword_Beta_Prof_la.EN_cc.India&campaigntype=Search&portfolio=India&language=EN&product=Course&test=&audience=Keyword&topic=Business_Analytics&priority=Beta&utm_content=deal4584&utm_term=_._ag_166539367088_._ad_696237424048_._kw_business%20analysis%20fundamentals_._de_c_._dm__._pl__._ti_kwd-489611425112_._li_9148661_._pd__._&matchtype=b&gad_source=1&gad_campaignid=21178772608&gbraid=0AAAAADROdO1W1X69Krd-IdkoDm5wYHdot&gclid=CjwKCAjwup3HBhAAEiwA7euZujLCrGZaYmVI9nwWGPdX9Ab7MY0UNDdpjl84j_ltSVKd-8UWPhy_choCojMQAvD_BwE", "reason": "Understand the role of a BA in projects."},
            {"title": "Excel for Business Analytics", "url": "https://www.coursera.org/learn/business-analytics-excel", "reason": "Analyze business data using Excel."},
            {"title": "Power BI Data Visualization", "url": "https://www.coursera.org/courses?query=microsoft%20power%20bi", "reason": "Visualize data effectively."},
            {"title": "SQL for Business Analysts", "url": "https://www.coursera.org/courses?query=sql", "reason": "Learn SQL queries for data extraction."},
            {"title": "Agile Project Management", "url": "https://www.coursera.org/courses?query=agile%20project%20management", "reason": "Adopt Agile methods in analysis."}
        ],
        "sap developer": [
            {"title": "SAP ABAP Programming", "url": "https://www.udemy.com/course/learn-sap-abap-sap-abap-programming-language-for-beginners/?srsltid=AfmBOooQYfn6Y-I7LS7ABffPm2772qPeh7_1mMpRrZ5McVoRSTv8Tn3m&couponCode=25BBPMXNVD35", "reason": "Learn ABAP for SAP development."},
            {"title": "SAP S/4HANA Overview", "url": "https://api.sap.com/products/SAPS4HANA/overview", "reason": "Understand S/4HANA fundamentals."},
            {"title": "SAP Fiori UI5 Development", "url": "https://www.udemy.com/course/professional-sapui5-web-application-development-part-1/?srsltid=AfmBOoqwTbBf0gFf9ZUY8SvPm8yFH4Ofmzt7v1CPX9dfCl4Sc3iBM6Av&couponCode=25BBPMXNVD35", "reason": "Develop SAP Fiori web apps."},
            {"title": "SAP Workflow Management", "url": "https://eursap.eu/blog/overview-of-sap-workflow-management", "reason": "Automate business processes with SAP."},
            {"title": "SAP Analytics Cloud", "url": "https://www.udemy.com/course/sap-analytics-cloud-sac/?srsltid=AfmBOopep0ZM8qc8JdE_KzVTF_WS57Vj0nwY7-Eu117Bw2wjiUkpzzza&couponCode=25BBPMXNVD35", "reason": "Visualize business insights using SAP tools."}
        ],
        "automation testing": [
            {"title": "Selenium WebDriver with Python", "url": "http://coursera.org/learn/selenium-webdriver-python?utm_medium=sem&utm_source=gg&utm_campaign=b2c_india_google-it-automation_google_ftcof_professional-certificates_cx_dr_bau_gg_pmax_pr_in_all_m_hyb_22-11_desktop&campaignid=19197733182&adgroupid=&device=c&keyword=&matchtype=&network=x&devicemodel=&creativeid=&assetgroupid=6458849661&targetid=&extensionid=&placement=&gad_source=1&gad_campaignid=19204410364&gbraid=0AAAAADdKX6av6S_WBTtgls7sWOPHCKzEQ&gclid=CjwKCAjwup3HBhAAEiwA7euZusyqeuY5k_j5XaoPDx0qjl-B80vYpFPLQ0frNaPLfmcoxJR3SObEVRoC4X8QAvD_BwE", "reason": "Automate web testing using Selenium."},
            {"title": "API Testing with Postman", "url": "https://www.udemy.com/course/postman-api-automation-testing-with-javascript/?im_ref=x380%3AaWl6xycWPoRPr1DozJvUkp3553VqW9SW00&sharedid=118678&irpid=1453307&utm_medium=affiliate&utm_source=impact&utm_audience=mx&utm_tactic=&utm_content=3281534&utm_campaign=1453307&irgwc=1&gad_source=1&couponCode=PMNVD2025", "reason": "Perform REST API automation."},
            {"title": "Cypress End-to-End Testing", "url": "https://www.udemy.com/course/cypress-end-to-end-testing-getting-started/?im_ref=x380%3AaWl6xycWPoRPr1DozJvUkp3553VqW9SW00&sharedid=118678&irpid=1453307&utm_medium=affiliate&utm_source=impact&utm_audience=mx&utm_tactic=&utm_content=3281534&utm_campaign=1453307&irgwc=1&gad_source=1", "reason": "Test modern web applications."},
            {"title": "Test Automation Frameworks", "url": "https://www.geeksforgeeks.org/software-testing/automation-testing-software-testing/", "reason": "Build maintainable test frameworks."},
            {"title": "CI/CD for Test Automation", "url": "https://www.udemy.com/course/cicd-testers/?im_ref=x380%3AaWl6xycWPoRPr1DozJvUkp3553VqW9SW00&sharedid=118678&irpid=1453307&utm_medium=affiliate&utm_source=impact&utm_audience=mx&utm_tactic=&utm_content=3281534&utm_campaign=1453307&irgwc=1&gad_source=1", "reason": "Integrate testing into CI/CD pipelines."}
        ],
        "electrical engineer": [
            {"title": "Electric Circuits", "url": "https://www.geeksforgeeks.org/physics/electric-circuit/", "reason": "Understand fundamental electrical systems."},
            {"title": "Embedded Systems", "url": "https://www.coursera.org/courses?query=embedded%20systems&index=prod_all_products_term_optimization_updated_ltv&userQuery=embedded%20systems&utm_medium=sem&utm_source=gg&utm_campaign=b2c_india_google-it-automation_google_ftcof_professional-certificates_cx_dr_bau_gg_pmax_pr_in_all_m_hyb_22-11_desktop&campaignid=19197733182&adgroupid=&device=c&keyword=&matchtype=&network=x&devicemodel=&creativeid=&assetgroupid=6458849661&targetid=&extensionid=&placement=&gad_source=1&gad_campaignid=19204410364&gbraid=0AAAAADdKX6av6S_WBTtgls7sWOPHCKzEQ&gclid=CjwKCAjwup3HBhAAEiwA7euZuuYthkhDUIY0cTMhvizqQLYvrhXm4G03pohF4PzOdDMZaRchkw1pUxoCjhQQAvD_BwE", "reason": "Learn microcontroller programming."},
            {"title": "Power Electronics", "url": "https://www.geeksforgeeks.org/electrical-engineering/power-electronics/", "reason": "Explore power conversion techniques."},
            {"title": "IoT for Electrical Engineers", "url": "https://www.coursera.org/learn/iot", "reason": "Connect devices using IoT systems."},
            {"title": "MATLAB for Engineers", "url": "https://www.coursera.org/specializations/matlab-programming-engineers-scientists", "reason": "Simulate circuits and signals."}
        ],
        "operations manager": [
            {"title": "Operations Management", "url": "https://www.coursera.org/learn/wharton-operations", "reason": "Learn supply chain and process management."},
            {"title": "Lean Six Sigma Green Belt", "url": "https://www.coursera.org/career-academy?utm_medium=sem&utm_source=gg&utm_campaign=b2c_india_x_multi_ftcof_career-academy_cx_dr_bau_gg_pmax_gc_in_all_m_hyb_24-03_desktop&campaignid=21104989118&adgroupid=&device=c&keyword=&matchtype=&network=x&devicemodel=&creativeid=&assetgroupid=6544944776&targetid=&extensionid=&placement=&gad_source=1&gad_campaignid=21104990591&gbraid=0AAAAADdKX6bDO2Frhr-obTf5aM1DEivN9&gclid=CjwKCAjwup3HBhAAEiwA7euZupVpcZdQxN9tbqn82BrO8fwqfw3n1ORDYgmQwrdAmo5bApv_xmFVlBoCj4AQAvD_BwE", "reason": "Enhance efficiency through Six Sigma."},
            {"title": "Supply Chain Management", "url": "https://www.coursera.org/specializations/supply-chain-management", "reason": "Optimize production and logistics."},
            {"title": "Project Management Principles", "url": "https://www.coursera.org/specializations/project-management", "reason": "Master project lifecycle and planning."},
            {"title": "Business Process Improvement", "url": "https://www.coursera.org/articles/business-process-improvement", "reason": "Streamline business operations."}
        ],
        "python developer": [
            {"title": "Complete Python Bootcamp", "url": "https://www.udemy.com/course/complete-python-bootcamp/?srsltid=AfmBOoomFKIzODWBRYliE9rtT1nL0ZSk2KFHqU4qvKqjrpe8bqatY6Zo", "reason": "Learn Python from basics to advanced."},
            {"title": "Flask & Django Web Development", "url": "https://www.udemy.com/course/python-and-flask-and-django-course-for-beginners/?srsltid=AfmBOop8V8OdRBXDhPQqvvjMPNA6l_M_Ngqm8RVsxGUzGFYQfIlfK9_9", "reason": "Develop full-stack web apps."},
            {"title": "Advanced Python Concepts", "url": "https://www.coursera.org/courses?query=python&productDifficultyLevel=Advanced", "reason": "Improve Python code performance."},
            {"title": "REST APIs in Python", "url": "https://www.coursera.org/learn/packt-rest-apis-with-flask-and-python-in-2024-i01az", "reason": "Build RESTful APIs using Flask."},
            {"title": "DSA in Python", "url": "udemy.com/course/data-structures-algorithms-in-python/?srsltid=AfmBOoqvDE8btSCb-gR4aepgAOC-CMgY4zbymUb0NGl7fRimMxtroMKO", "reason": "Prepare for coding interviews."}
        ],
        "devops engineer": [
            {"title": "Docker and Kubernetes", "url": "https://www.coursera.org/learn/ibm-containers-docker-kubernetes-openshift", "reason": "Manage containerized apps."},
            {"title": "DevOps Foundations", "url": "https://www.udemy.com/course/devops-foundation-j/?srsltid=AfmBOoqDKz7x80SFJ8avpVKNkGKBizlKvQSQE_Q9P4CVXzI7-He-qRTd&couponCode=PMNVD2025", "reason": "Understand CI/CD workflows."},
            {"title": "Terraform Infrastructure", "url": "https://www.datacamp.com/tutorial/getting-started-terraform", "reason": "Automate infrastructure deployment."},
            {"title": "AWS DevOps Certification", "url": "https://aws.amazon.com/certification/certified-devops-engineer-professional/", "reason": "Prepare for AWS DevOps certification."},
            {"title": "Jenkins for DevOps", "url": "https://medium.com/@mesagarkulkarni/devops-tool-jenkins-ci-cd-a942b7b53876", "reason": "Automate builds with Jenkins."}
        ],
        "network security engineer": [
            {"title": "Network Security Fundamentals", "url": "https://www.coursera.org/courses?query=network%20security", "reason": "Understand secure network protocols."},
            {"title": "Ethical Hacking", "url": "https://www.coursera.org/courses?query=ethical%20hacking", "reason": "Learn ethical hacking and penetration testing."},
            {"title": "Cybersecurity for Cloud", "url": "https://aws.amazon.com/free/?all-free-tier.sort-by=item.additionalFields.SortRank&all-free-tier.sort-order=asc&awsf.Free%20Tier%20Types=*all&awsf.Free%20Tier%20Categories=categories%23compute&trk=11f5779c-ce0f-4a8d-986e-40191b2182bb&sc_channel=ps&ef_id=CjwKCAjwup3HBhAAEiwA7euZulqCzYXRH7pP3wGjxzAtyfEih1bFVwrnZ3FnkjRADas_XbMyNTcmYBoCVkkQAvD_BwE:G:s&s_kwcid=AL!4422!3!770488444505!p!!g!!cloud%20service!22918816463!185733392602&gad_campaignid=22918816463&gbraid=0AAAAADjHtp9gcObZV6qfT5jGuJ6yDrZ6d&gclid=CjwKCAjwup3HBhAAEiwA7euZulqCzYXRH7pP3wGjxzAtyfEih1bFVwrnZ3FnkjRADas_XbMyNTcmYBoCVkkQAvD_BwE", "reason": "Protect cloud-based systems."},
            {"title": "Firewalls and VPNs", "url": "https://www.geeksforgeeks.org/ethical-hacking/relationship-between-vpn-and-firewall/", "reason": "Secure networks with firewalls."},
            {"title": "CompTIA Security+", "url": "https://www.coursera.org/in/articles/what-is-the-comptia-security-plus-certification", "reason": "Earn recognized security certification."}
        ],
        "pmo": [
            {"title": "Project Management Fundamentals", "url": "https://www.coursera.org/learn/project-management-foundations", "reason": "Learn project planning and execution."},
            {"title": "Agile and Scrum", "url": "https://www.coursera.org/learn/agile-development-and-scrum", "reason": "Master Agile methodologies."},
            {"title": "Risk Management in Projects", "url": "https://www.coursera.org/in/articles/how-to-manage-project-risk", "reason": "Identify and mitigate project risks."},
            {"title": "MS Project for PMO", "url": "https://www.coursera.org/professional-certificates/microsoft-project-management", "reason": "Plan efficiently using MS Project."},
            {"title": "Strategic Project Management", "url": "https://www.coursera.org/courses?query=project%20management", "reason": "Align project goals with business strategy."}
        ],
        "database": [
            {"title": "SQL for Beginners", "url": "https://www.coursera.org/courses?query=sql", "reason": "Master SQL basics for databases."},
            {"title": "Database Design", "url": "https://www.coursera.org/courses?query=database%20design", "reason": "Understand relational models and normalization."},
            {"title": "MySQL Advanced", "url": "https://www.coursera.org/learn/advanced-mysql-topics", "reason": "Write complex queries efficiently."},
            {"title": "MongoDB Developer", "url": "https://www.udemy.com/course/mongodb-the-complete-developers-guide/?srsltid=AfmBOoqP47Wp40QHFKeFkeKC9c0miovyNff6cwNF0R8Tb85kD92oG-bv&couponCode=PMNVD2025", "reason": "Learn NoSQL and document-oriented DBs."},
            {"title": "PostgreSQL for Developers", "url": "https://aws.amazon.com/free/database/?trk=0ae53ae1-71bb-4631-a295-3c7f2f60b821&sc_channel=ps&ef_id=CjwKCAjwup3HBhAAEiwA7euZuqAIEgR47Ur8A357GDAqs4NX1VblqLqF6KhW_ekJF-LOJ59UUpyZ8BoC03cQAvD_BwE:G:s&s_kwcid=AL!4422!3!770402408608!e!!g!!postgres!22913054811!189888864131&gad_campaignid=22913054811&gbraid=0AAAAADjHtp8if1AObfzhyZ4sDi-rh3v3i&gclid=CjwKCAjwup3HBhAAEiwA7euZuqAIEgR47Ur8A357GDAqs4NX1VblqLqF6KhW_ekJF-LOJ59UUpyZ8BoC03cQAvD_BwE", "reason": "Master advanced SQL queries in PostgreSQL."}
        ],
        "hadoop": [
            {"title": "Big Data Hadoop Fundamentals", "url": "https://www.coursera.org/courses?query=hadoop", "reason": "Understand Hadoop architecture."},
            {"title": "HDFS and MapReduce", "url": "https://aws.amazon.com/pm/emr/?trk=54301fc3-5516-44bc-90ed-128834533769&sc_channel=ps&ef_id=CjwKCAjwup3HBhAAEiwA7euZusBRVw6VFpMXbajhiwnutJlu45M_1GpLXj-0SNdatyWfJKjD88QkxhoCgN4QAvD_BwE:G:s&s_kwcid=AL!4422!3!770402756833!p!!g!!mapreduce!22913055969!190996813904&gad_campaignid=22913055969&gbraid=0AAAAADjHtp8xvlkTwK_tGoF6E352CxzTt&gclid=CjwKCAjwup3HBhAAEiwA7euZusBRVw6VFpMXbajhiwnutJlu45M_1GpLXj-0SNdatyWfJKjD88QkxhoCgN4QAvD_BwE", "reason": "Learn distributed data processing."},
            {"title": "Apache Spark with Hadoop", "url": "https://www.coursera.org/learn/introduction-to-big-data-with-spark-hadoop", "reason": "Integrate Spark with Hadoop for analytics."},
            {"title": "Hive and Pig Scripting", "url": "https://medium.com/@noel.benji/integrating-apache-pig-with-hadoop-ecosystem-tools-a-complete-guide-8504356c77fb", "reason": "Process data efficiently using Hive."},
            {"title": "Data Engineering with Hadoop", "url": "https://www.coursera.org/professional-certificates/ibm-data-engineer", "reason": "Build big data pipelines using Hadoop ecosystem."}
        ],
        "etl developer": [
            {"title": "ETL Concepts for Beginners", "url": "https://www.coursera.org/courses?query=etl", "reason": "Understand ETL fundamentals."},
            {"title": "Informatica PowerCenter", "url": "https://www.udemy.com/course/informatica-powercenter-961-beginners-to-advanced/?srsltid=AfmBOoqGSaPRscmW9SyPWnjylXbxOBXZzthADtRAdGqowY9-jY4UJ7Mc&couponCode=PMNVD2025", "reason": "Master data integration workflows."},
            {"title": "Data Warehousing", "url": "https://www.coursera.org/specializations/data-warehousing", "reason": "Design and manage data warehouses."},
            {"title": "SQL for ETL Developers", "url": "https://www.rudderstack.com/learn/etl/etl-and-sql-how-they-work-together/", "reason": "Write efficient ETL SQL scripts."},
            {"title": "Apache Airflow for Data Pipelines", "url": "https://www.coursera.org/learn/etl-and-data-pipelines-shell-airflow-kafka", "reason": "Automate data pipelines using Airflow."}
        ],
        "dotnet developer": [
            {"title": ".NET Core for Beginners", "url": "https://www.coursera.org/learn/intro-to-dotnet-core", "reason": "Learn C# and .NET Core fundamentals."},
            {"title": "ASP.NET MVC", "url": "https://www.udemy.com/topic/aspnet-mvc/?srsltid=AfmBOoo06jwCvorijadbuEyah5ToSXfKss0B_zyBVtxiDkHlBePTRx-Y", "reason": "Develop dynamic web apps."},
            {"title": "Entity Framework Core", "url": "https://medium.com/@ravipatel.it/a-beginners-guide-to-entity-framework-core-ef-core-5cde48fc7f7a", "reason": "Master ORM and database connectivity."},
            {"title": "Azure for .NET Developers", "url": "https://www.coursera.org/learn/packt-microsoft-azure-for-net-developers-ah6so", "reason": "Deploy .NET apps to Azure."},
            {"title": "REST API with .NET", "url": "https://dotnet.microsoft.com/en-us/apps/aspnet/apis", "reason": "Build APIs using ASP.NET Core."}
        ],
        "blockchain": [
            {"title": "Blockchain Basics", "url": "https://www.coursera.org/learn/blockchain-basics", "reason": "Understand blockchain fundamentals."},
            {"title": "Ethereum Smart Contracts", "url": "https://www.geeksforgeeks.org/solidity/smart-contracts-in-blockchain/", "reason": "Develop smart contracts using Solidity."},
            {"title": "Hyperledger Fabric", "url": "https://www.geeksforgeeks.org/computer-networks/hyperledger-fabric-in-blockchain/", "reason": "Explore enterprise blockchain networks."},
            {"title": "DeFi and Web3", "url": "https://lemon.io/answers/web3/what-is-the-role-of-web3-in-decentralized-finance-defi/", "reason": "Learn decentralized finance fundamentals."},
            {"title": "Crypto Security", "url": "https://www.arkoselabs.com/explained/guide-to-cryptocurrency-security/", "reason": "Understand cryptography in blockchain."}
        ],
        "testing": [
            {"title": "Software Testing Fundamentals", "url": "https://www.coursera.org/courses?query=software%20testing", "reason": "Learn testing techniques and principles."},
            {"title": "Manual Testing Bootcamp", "url": "https://www.udemy.com/course/testerbootcamp/?srsltid=AfmBOooJUFLTifImp9eSwe8DEVf162dXfmGj1jGLeKFf2iDt0SxnICDa/", "reason": "Understand SDLC and STLC."},
            {"title": "API Testing with Postman", "url": "https://www.udemy.com/course/postman-api-automation-testing-with-javascript/?im_ref=x380%3AaWl6xycWPoRPr1DozJvUkp3553VqW9SW00&sharedid=118678&irpid=1453307&utm_medium=affiliate&utm_source=impact&utm_audience=mx&utm_tactic=&utm_content=3281534&utm_campaign=1453307&irgwc=1&gad_source=1&couponCode=PMNVD2025", "reason": "Test APIs efficiently."},
            {"title": "Automation Testing with Selenium", "url": "https://www.coursera.org/courses?query=selenium", "reason": "Automate functional testing."},
            {"title": "Agile Testing", "url": "https://www.coursera.org/courses?query=agile%20testing", "reason": "Adapt QA practices in Agile environments."}
        ]
    }
    prompt = f"""
    The user has completed a mock interview for the role of {role_name}.
    Their average interview confidence score was {avg_confidence:.2f} out of 1.0.
    Recommend 5 recent online resources (courses, blogs, or YouTube channels)
    that can help improve their skills in this role.
    Return strictly as JSON list of objects:
    [{{"title": "...", "url": "...", "reason": "..."}}]
    """
    try:
        text = call_gemini(prompt, timeout=15) or ""
        text = text.strip()
        resources = json.loads(text)
        if not isinstance(resources, list) or len(resources) < 5:
            raise ValueError("Incomplete Gemini data")
        return resources[:5]
    except Exception:
        return role_courses.get(role_name, [
            {"title": "Explore Roadmap.sh", "url": "https://roadmap.sh/", "reason": "Role-based learning roadmaps."},
            {"title": "LinkedIn Learning", "url": "https://www.linkedin.com/learning/", "reason": "Professional upskilling resources."},
            {"title": "Coursera", "url": "https://www.coursera.org/", "reason": "Global university-level courses."},
            {"title": "Udemy", "url": "https://www.udemy.com/", "reason": "Affordable practical online courses."},
            {"title": "YouTube: FreeCodeCamp", "url": "https://www.youtube.com/@freecodecamp", "reason": "Free tutorials and coding guides."}
        ])
def get_db_connection():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='dhivya@2005',
        database='mockmate_db'
    )
@app.route('/submit_score', methods=['POST'])
def submit_score():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "message": "No JSON data received"}), 400
        avg_conf = float(data.get('avg_conf', 0))
        avg_emotion = float(data.get('avg_emotion', 0))
        roles = session.get('predicted_roles', [])
        role_name = roles[0]['role'] if roles else "Candidate"
        resources = fetch_ai_learning_resources(role_name, avg_conf)
        user_id = session.get('user_id')
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO interview_scores 
            (user_id, avg_confidence, avg_emotion_score, job_role, learning_resources) 
            VALUES (%s, %s, %s, %s, %s)
            """,
            (user_id, avg_conf, avg_emotion, role_name, json.dumps(resources))
        )
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({"success": True, "redirect": url_for("dashboard")})
    except Exception as e:
        print("Error saving score:", e)
        return jsonify({"success": False, "message": str(e)}), 500
@app.route('/dashboard')
def dashboard():
    user_id = session.get('user_id')  # Logged-in user
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)
    cursor.execute("""
        SELECT id, avg_confidence, avg_emotion_score, job_role, learning_resources, created_at
        FROM interview_scores
        WHERE user_id = %s
        ORDER BY created_at ASC
    """, (user_id,))
    results = cursor.fetchall()
    cursor.close(); db.close()

    # Prepare chart data
    labels = [row['created_at'].strftime("%Y-%m-%d %H:%M") for row in results]
    confidence_scores = [float(row['avg_confidence']) for row in results]
    emotion_scores = [float(row['avg_emotion_score']) for row in results]

    # Parse learning resources JSON
    for row in results:
        try:
            row['learning_resources'] = json.loads(row.get('learning_resources') or "[]")
        except Exception:
            row['learning_resources'] = []

    return render_template('dashboard.html',
                           chart_data={"labels": labels, "confidence": confidence_scores, "emotion": emotion_scores},
                           results=results)
if __name__ == '__main__':
    # ensure uploads dir
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    app.run(debug=True, host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
