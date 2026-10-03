import streamlit as st
import tensorflow as tf
import numpy as np
import pickle

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Medical Text Predictor",
    page_icon="🩺",
    layout="centered"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }

    .title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .subtitle {
        text-align: center;
        color: #6b7280;
        font-size: 17px;
        margin-bottom: 30px;
    }

    .result-box {
        background-color: #f5f7fa;
        border-left: 5px solid #2563eb;
        padding: 20px;
        border-radius: 8px;
        margin-top: 20px;
        line-height: 1.7;
        font-size: 16px;
    }

    .info-box {
        background-color: #eef6ff;
        padding: 14px 18px;
        border-radius: 8px;
        margin-top: 20px;
        font-size: 14px;
    }

    .footer {
        text-align: center;
        color: #888;
        font-size: 13px;
        margin-top: 35px;
    }
</style>
""", unsafe_allow_html=True)


# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("medical_lstm_epoch200.keras")


@st.cache_resource
def load_tokenizer():
    with open("medical_tokenizer.pkl", "rb") as f:
        return pickle.load(f)


model = load_model()
tokenizer = load_tokenizer()

SEQ_LENGTH = 20


# =========================
# GENERATE MEDICAL TEXT
# =========================
def generate_medical_text(seed_text, next_words):

    generated_text = seed_text.lower()

    for _ in range(next_words):

        token_list = tokenizer.texts_to_sequences(
            [generated_text]
        )[0]

        token_list = token_list[-SEQ_LENGTH:]

        token_list = tf.keras.preprocessing.sequence.pad_sequences(
            [token_list],
            maxlen=SEQ_LENGTH,
            padding="pre"
        )

        predicted = model.predict(
            token_list,
            verbose=0
        )

        predicted_word_index = np.argmax(
            predicted,
            axis=-1
        )[0]

        output_word = ""

        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                output_word = word
                break

        if output_word == "":
            break

        generated_text += " " + output_word

    return generated_text


# =========================
# HEADER
# =========================
st.markdown(
    '<div class="title">Medical Text Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'LSTM-based next-word prediction trained on medical transcriptions'
    '</div>',
    unsafe_allow_html=True
)


# =========================
# INPUT
# =========================
st.subheader("Enter Medical Text")

seed_text = st.text_area(
    "Starting text",
    value="the patient was admitted with",
    height=120,
    placeholder="Enter a medical sentence..."
)

next_words = st.slider(
    "Number of words to generate",
    min_value=1,
    max_value=50,
    value=15
)


# =========================
# GENERATE
# =========================
if st.button(
    "Generate Medical Text",
    use_container_width=True
):

    if not seed_text.strip():

        st.warning("Please enter some medical text.")

    else:

        with st.spinner("Generating text..."):

            result = generate_medical_text(
                seed_text,
                next_words
            )

        st.subheader("Generated Text")

        st.markdown(
            f'<div class="result-box">{result}</div>',
            unsafe_allow_html=True
        )

        st.success(
            f"Generated {next_words} additional words."
        )


# =========================
# MODEL INFORMATION
# =========================
st.markdown(
    """
    <div class="info-box">
        <b>Model:</b> LSTM<br>
        <b>Sequence Length:</b> 20 words<br>
        <b>Training:</b> Medical Transcriptions Dataset<br>
        <b>Output:</b> Medical-domain next-word generation
    </div>
    """,
    unsafe_allow_html=True
)


# =========================
# DISCLAIMER
# =========================
st.markdown(
    """
    <div class="footer">
        This application is for educational and research purposes only.
        Generated text should not be used for medical diagnosis or treatment.
    </div>
    """,
    unsafe_allow_html=True
)