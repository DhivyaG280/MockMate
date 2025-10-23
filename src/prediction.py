import torch
import fitz  # PyMuPDF
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch.nn.functional as F
import sys

# === Load model and tokenizer ===
model_path = "job_role_model"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
model.eval()

# Load label mapping
id2label = torch.load("job_role_model/id2label.pt")


# === Function to extract text from PDF ===
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text.strip()


# === Function to predict top 3 job roles ===
def predict_top_3_roles(resume_text):
    inputs = tokenizer(resume_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)
    top3_probs, top3_indices = torch.topk(probs, k=3)

    top_roles = []
    for idx, prob in zip(top3_indices[0], top3_probs[0]):
        role = id2label[idx.item()]
        confidence = round(prob.item(), 4)
        top_roles.append((role, confidence))

    return top_roles


# === Main Execution ===
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict_from_pdf.py <path_to_resume.pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    resume_text = extract_text_from_pdf(pdf_path)

    if not resume_text:
        print("Error: No text found in PDF.")
        sys.exit(1)

    predictions = predict_top_3_roles(resume_text)
    print("\nTop 3 Job Role Predictions:")
    for role, score in predictions:
        print(f"{role}: {score * 100:.2f}%")
