"""
Streamlit front-end for the AI-Powered Crop Recommendation System.
Run:  streamlit run app.py
"""
import pandas as pd
import streamlit as st

import config as cfg
from predict import load_artefact, recommend, validate_inputs

st.set_page_config(page_title="Crop Recommendation", page_icon="🌾", layout="wide")


@st.cache_resource
def get_artefact():
    return load_artefact()


st.title("🌾 AI-Powered Crop Recommendation System")
st.caption("Data-driven crop selection using soil nutrients and environmental conditions")

try:
    artefact = get_artefact()
except FileNotFoundError:
    st.error("Model file not found. Please run `python train.py` first.")
    st.stop()

with st.sidebar:
    st.header("About")
    st.write(f"**Model:** {artefact['model_name']} ({artefact['version']})")
    st.write(f"**Test accuracy:** {artefact['metrics']['accuracy']:.2%}")
    st.write(f"**Macro F1:** {artefact['metrics']['macro_f1']:.3f}")
    st.write(f"**Crops supported:** {len(artefact['classes'])}")
    st.divider()
    st.subheader("Project Team")
    for name, roll in cfg.AUTHORS:
        st.write(f"**{name}**  \nRoll No: {roll}")

st.subheader("Enter field conditions")
cols = st.columns(4)
values = {}
for i, (feat, (lo, hi, default, step, label)) in enumerate(cfg.FEATURE_INFO.items()):
    with cols[i % 4]:
        values[feat] = st.number_input(label, min_value=lo, max_value=hi,
                                       value=default, step=step)

top_k = st.slider("Number of alternative crops to show", 1, 5, 3)

if st.button("Generate Recommendation", type="primary"):
    errors = validate_inputs(values)
    if errors:
        for e in errors:
            st.error(e)
    else:
        ranked = recommend(values, top_k=top_k, artefact=artefact)
        best_crop, best_p = ranked[0]
        st.success(f"Recommended crop: **{best_crop.upper()}**  (confidence {best_p:.1%})")
        st.write("#### Top recommendations")
        df = pd.DataFrame(ranked, columns=["Crop", "Probability"]).set_index("Crop")
        st.bar_chart(df)
        st.dataframe(df.style.format("{:.1%}"), use_container_width=True)

st.info("Disclaimer: this is a decision-support tool. Please verify the recommendation "
        "with local agronomic expertise, seasonality and market conditions.")
