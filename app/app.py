import streamlit as st

st.set_page_config(layout="wide")

pages = [
    st.Page("app/pages/biography.py", title="Biography"),
    st.Page("app/pages/1_resume.py", title="Resume"),
    st.Page("app/pages/2_Projects_Overview.py", title="General Projects"),
    st.Page("app/pages/3_ICU_Triage_Project.py", title="ICU Triage Project")
]

pg = st.navigation(pages)
pg.run()