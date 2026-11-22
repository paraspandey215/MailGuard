
import streamlit as st
import joblib
import pandas as pd
from pathlib import Path
from preprocessing_v2 import extract_features

st.set_page_config(
    page_title="MailGuard | Email Security",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="collapsed"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "spam_phishing_model_v2.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #071426 0%, #0b1930 55%, #10243d 100%);
    color: #e2e8f0;
}

.block-container {
    max-width: 900px;
    padding: 2.5rem 1.5rem 2rem;
}

html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

p, label, .stMarkdown {
    color: #dbeafe;
    font-size: 19px;
}

.hero {
    background: linear-gradient(120deg, #0c1d35, #123b6);
    border: 1px solid #24547c;
    padding: 20px 5px;
    border-radius: 22px;
    text-align: center;
    margin-bottom: 15px;
    box-shadow: 0 12px 35px rgba(0, 0, 0, 0.25);
}

.hero h1 {
    color: #f8fafc;
    font-size: clamp(27px, 4vw, 34px);
    font-weight: 800;
    letter-spacing: -1px;
    margin: 0;
}

.hero p {
    color: #bae6fd;
    font-size: 17px;
    margin-top: 8px;
}

.security-badge {
    display: inline-block;
    background: rgba(56, 189, 248, 0.12);
    border: 1px solid #38bdf8;
    color: #7dd3fc;
    padding: 8px 15px;
    border-radius: 30px;
    font-size: 14px;
    font-weight: 700;
    margin-bottom: 15px;
    letter-spacing: 1px;
}

.section-title {
    color: #f8fafc;
    font-size: 25px;
    font-weight: 750;
    margin-bottom: 8px;
}

.helper {
    color: #a9bdd3;
    font-size: 19px;
    line-height: 1.6;
    margin-bottom: 22px;
}

div[data-testid="stForm"] {
    background: #10243b;
    border: 1px solid #294761;
    border-radius: 18px;
    padding: 25px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
}

div[data-testid="stTextInput"] label,
div[data-testid="stTextArea"] label {
    color: #e0f2fe !important;
    font-size: 19px !important;
    font-weight: 650 !important;
}

div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea {
    background: #07182b !important;
    color: #f8fafc !important;
    border: 1px solid #365773 !important;
    border-radius: 10px !important;
    font-size: 19px !important;
    padding: 12px !important;
}

div[data-testid="stTextInput"] input::placeholder,
div[data-testid="stTextArea"] textarea::placeholder {
    color: #829bb3 !important;
    font-size: 19px !important;
}

div[data-testid="stTextInput"] input:focus,
div[data-testid="stTextArea"] textarea:focus {
    border: 1px solid #38bdf8 !important;
    box-shadow: 0 0 0 2px rgba(56,189,248,0.15) !important;
}

div.stFormSubmitButton > button {
    background: linear-gradient(90deg, #0284c7, #38bdf8);
    color: #061526 !important;
    border: none;
    border-radius: 11px;
    min-height: 54px;
    font-size: 24px;
    font-weight: 800;
    transition: none !important;
    animation: none !important;
}

div.stFormSubmitButton > button:hover {
    background: linear-gradient(90deg, #0284c7, #38bdf8);
    transform: none !important;
    box-shadow: none !important;
}

div.stFormSubmitButton > button:active {
    transform: none !important;
    box-shadow: none !important;
}

div[data-testid="stDialog"] {
    background: #0c1d32;
}

div[data-testid="stDialog"] > div {
    background: #0c1d32;
    border: 1px solid #315674;
    border-radius: 18px;
}

div[data-testid="stDialog"] h2 {
    color: #f8fafc;
    font-size: 28px;
}

.result-card {
    padding: 30px 20px;
    border-radius: 16px;
    text-align: center;
    margin: 12px 0 20px;
}

.result-card h2 {
    font-size: 30px;
    font-weight: 800;
    margin: 12px 0;
}

.result-card p {
    font-size: 19px;
    line-height: 1.6;
    margin-bottom: 0;
}

.safe {
    background: #062c2a;
    border: 1px solid #10b981;
    color: #6ee7b7;
}

.danger {
    background: #35151e;
    border: 1px solid #fb7185;
    color: #fda4af;
}

div[data-testid="stDialog"] div.stButton > button {
    background: #1e354c;
    color: #e0f2fe;
    border: 1px solid #45617a;
    border-radius: 9px;
    min-height: 42px;
    font-size: 19px;
    font-weight: 700;
}

div[data-testid="stDialog"] div.stButton > button:hover {
    background: #294761;
    color: white;
    border-color: #38bdf8;
}

div[data-testid="stAlert"] p {
    color: #000000 !important;
    font-size: 19px !important;
}

.footer {
    text-align: center;
    color: #829bb3;
    font-size: 17px;
    margin-top: 35px;
    padding: 20px 0;
    border-top: 1px solid #294761;
}

@media (max-width: 600px) {
    .block-container {
        padding: 1.2rem 0.8rem;
    }

    .hero {
        padding: 30px 15px;
        border-radius: 16px;
    }

    div[data-testid="stForm"] {
        padding: 16px;
    }

    .section-title {
        font-size: 22px;
    }
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>📧 MailGuard</h1>
    <p>Smart Email Spam & Phishing Detection</p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="section-title">Analyze Your Email</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="helper">Check your email for suspicious content. '
    'Enter the subject and message below to get an instant analysis.</div>',
    unsafe_allow_html=True
)

with st.form("email_form"):
    subject = st.text_input(
        "Email Subject",
        placeholder="e.g. Important account notification"
    )

    body = st.text_area(
        "Email Message",
        placeholder="Paste your complete email message here...",
        height=210
    )

    submitted = st.form_submit_button(
        "🔍  Analyze Email",
        use_container_width=True
    )

@st.dialog("Email Security Analysis", dismissible=True)
def show_result_popup(is_spam):
    if is_spam:
        st.markdown("""
        <div class="result-card danger">
            <div style="font-size:55px;">⚠️</div>
            <h2>Spam / Phishing Detected</h2>
            <p>This email has been classified as potentially suspicious.
            Avoid clicking links or downloading attachments.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-card safe">
            <div style="font-size:55px;">✓</div>
            <h2>Ham / Legitimate Email</h2>
            <p>This email has been classified as legitimate
            by the MailGuard detection model.</p>
        </div>
        """, unsafe_allow_html=True)

    st.caption(
        "AI predictions may not always be correct. "
        "Always verify suspicious emails carefully."
    )

    if st.button("✕  Close Result", use_container_width=True):
        st.rerun()

if submitted:
    if not subject.strip() and not body.strip():
        st.warning("Please enter an email subject or message.")
    else:
        email = f"Subject: {subject}\n{body}"

        try:
            features = extract_features(email)
            input_df = pd.DataFrame([features])

            columns = [
                "clean_text",
                "subject_length",
                "character_count",
                "word_count",
                "url_count",
                "email_count",
                "shortener_count",
                "ip_url_count",
                "exclamation_count",
                "question_count",
                "digit_ratio",
                "uppercase_ratio",
                "has_html"
            ]

            prediction = model.predict(input_df[columns])[0]
            show_result_popup(prediction == 1)

        except Exception as e:
            st.error(f"Analysis failed: {e}")

st.markdown("""
<div class="footer">
    <strong style="color:#bae6fd;">MailGuard</strong><br>
    Email Spam & Phishing Detection System<br>
    Powered by Python · Scikit-learn · Streamlit
</div>
""", unsafe_allow_html=True)
