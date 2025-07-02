# Home.py
import streamlit as st

st.set_page_config(page_title="🌊 Coral Reef AI Models", page_icon="🐠", layout="centered")
st.title("🌊 Coral Reef AI Models")
st.markdown("""
Welcome! Explore different coral reef simulation & prediction tools.
""")

# CSS for cards
st.markdown("""
<style>
.card {
    background-color: #f0f8ff;
    padding: 20px;
    margin: 10px;
    border-radius: 10px;
    text-align: center;
    box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
}
.card-title { font-size: 22px; margin-bottom: 10px; }
.card-desc { font-size: 16px; color: #444; }
</style>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="card">
        <div class="card-title">🔧 Mechanistic ODE Model</div>
        <div class="card-desc">Simulate coral, algae & herbivore dynamics.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open ODE Model"):
        st.switch_page("pages/coral_app.py")

with c2:
    st.markdown("""
    <div class="card">
        <div class="card-title">🤖 NN Predict Final Coral</div>
        <div class="card-desc">ML model that predicts only the final coral cover.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open NN Final Coral"):
        st.switch_page("pages/coral_nn_webapp.py")

with c3:
    st.markdown("""
    <div class="card">
        <div class="card-title">📈 NN Predict Full Time Series</div>
        <div class="card-desc">ML model predicting coral time series curve.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open NN Full Series"):
        st.switch_page("pages/coral_nn_webapp.py")
