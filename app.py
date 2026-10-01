import streamlit as st
import numpy as np
import pickle
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Medical Text Predictor",
    page_icon="🩺",
    layout="centered"
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 38px;
            font-weight: 700;
            margin-bottom: 5px;
        }

        .subtitle {
            font-size: 17px;
            color: #666;
            margin-bottom: 25px;
        }

        .prediction-box {
            padding: 20px;
            border-radius: 12px;
            background-color: #f4f7fb;
            border: 1px solid #d9e1ec;
            margin-top: 20px;
            line-height: 1.7;
            font-size: 17px;
        }

        .info-box {
            padding: 15px;
            border-radius: 10px;
            background-color: #f8f9fa;
            border-left: 4px solid #4c78a8;
            margin-top: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_lstm_model():

    model = load_model(
        "medical_lstm_epoch100.keras"
    )

    return model


# ============================================================
# LOAD TOKENIZER
# ============================================================

@st.cache_resource
def load_tokenizer():

    with open(
        "medical_tokenizer.pkl",
        "rb"
    ) as f:

        tokenizer = pickle.load(f)

    return tokenizer


# ============================================================
# LOAD MODEL + TOKENIZER
# ============================================================

try:

    model = load_lstm_model()
    tokenizer = load_tokenizer()

except Exception as e:

    st.error(
        "Unable to load the model or tokenizer."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# SETTINGS
# ============================================================

SEQ_LENGTH = 20


# ============================================================
# NEXT WORD PREDICTION
# ============================================================

def predict_next_word(seed_text):

    token_list = tokenizer.texts_to_sequences(
        [seed_text]
    )[0]

    token_list = token_list[-SEQ_LENGTH:]

    token_list = pad_sequences(
        [token_list],
        maxlen=SEQ_LENGTH,
        padding="pre"
    )

    prediction = model.predict(
        token_list,
        verbose=0
    )

    predicted_index = np.argmax(
        prediction,
        axis=-1
    )[0]

    for word, index in tokenizer.word_index.items():

        if index == predicted_index:

            return word

    return ""


# ============================================================
# GENERATE MEDICAL TEXT
# ============================================================

def generate_medical_text(
    seed_text,
    next_words
):

    generated_text = seed_text.strip()

    for _ in range(next_words):

        next_word = predict_next_word(
            generated_text
        )

        if not next_word:
            break

        generated_text += " " + next_word

    return generated_text


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">Medical Text Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'LSTM-based next-word prediction using medical transcription data'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# INPUT
# ============================================================

st.subheader("Enter Medical Text")

seed_text = st.text_area(
    "Starting text",
    value="the patient was admitted with",
    height=120,
    placeholder="Enter a medical sentence or phrase..."
)


# ============================================================
# WORD COUNT
# ============================================================

next_words = st.slider(
    "Words to generate",
    min_value=1,
    max_value=50,
    value=20
)


# ============================================================
# GENERATE BUTTON
# ============================================================

if st.button(
    "Generate Medical Text",
    use_container_width=True
):

    if not seed_text.strip():

        st.warning(
            "Please enter some starting text."
        )

    else:

        with st.spinner(
            "Generating text..."
        ):

            result = generate_medical_text(
                seed_text,
                next_words
            )

        st.subheader(
            "Generated Text"
        )

        st.markdown(
            f"""
            <div class="prediction-box">
                {result}
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# INFORMATION
# ============================================================

st.markdown(
    """
    <div class="info-box">
        <strong>About this model</strong><br><br>
        This application uses an LSTM language model trained on
        medical transcription text to predict the next words in
        a given text sequence.
        <br><br>
        The generated text is model output and should not be used
        for medical diagnosis, treatment, or clinical decision-making.
    </div>
    """,
    unsafe_allow_html=True
)