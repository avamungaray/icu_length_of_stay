import streamlit as st

# Layout for top-of-page elements (title + headshot)
header_left, header_right = st.columns([4, 1], vertical_alignment="center")

# Title and Resume Download Button
with header_left:
    st.title("Ava Mungaray Resume")
    st.subheader("Data Scientist | Marketing Analytics Specialist")
    
    # with open("Ava_Mungaray_Resume_2026.pdf", "rb") as pdf_file:
        # st.download_button(
            # label="Download Full PDF Resume",
            # data=pdf_file,
            # file_name="Ava_Mungaray_Resume.pdf",
            # mime="application/pdf"
        # )

# Headshot
with header_right:
    st.image("pages/IMG_6877.jpeg", use_container_width=True)
st.markdown("---")

# Split into 2 columns
col1, col2 = st.columns([1, 1], gap="large")

# Technical Skills & Education
with col1:
    
    st.header("Technical Skills")
    
    st.subheader("Programming Languages")
    st.markdown("- **Python:** Advanced Core (Logic, Functions, Flow Control), IPython, Jupyter Ecosystem")
    st.markdown("- **R:** Statistical Modeling, Advanced Regression, Applied Classification")
    st.markdown("- **SQL:** Relational Database Design, Schema Implementation, Data Organization")
    
    st.subheader("Machine Learning & Modeling")
    st.markdown("- **Supervised Learning:** Linear, Multiple & Logistic Regression, Support Vector Machines (SVM)")
    st.markdown("- **Deep Learning:** Neural Networks, Deep Learning Architectures, Sequential/Time-Series Modeling")
    st.markdown("- **Methodologies:** Supervised/Unsupervised Landscapes, Model Selection, Hyperparameter Tuning & Optimization")
    
    st.subheader("Data Engineering & Analytics")
    st.markdown("- **Manipulation & Pipeline:** NumPy, Pandas (Advanced Data Loading, Aggregating, Grouping, and Cleaning)")
    st.markdown("- **Acquisition:** Data Sourcing, Processing, ETL Foundations, Database Management")

    st.subheader("Data Storytelling & BI")
    st.markdown("- **Tableau:** Interactive Enterprise Dashboarding, Actionable KPI Tracking")
    st.markdown("- **Python Visualization:** Matplotlib, Seaborn Analytics")
    st.markdown("- **Frameworks:** Streamlit Data App Deployment")

    st.subheader("Education")
    st.markdown("**M.S. in Data Science**")
    st.markdown("*Eastern University, 2026*")
    st.markdown("**B.A. in Public Relations**")
    st.markdown("*Pepperdine University, 2021*")

# Professional Experience
with col2:
    st.header("Professional Experience")
    
    # Experience 1
    st.markdown("### **Senior Digital Marketing Strategist**")
    st.markdown("*EducationDynamics* | *2023 - Present*")
    st.markdown(
        "- **Marketing Strategy:** Organize, create, and execute digital marketing efforts to generate leads and enrollments for universities."
    )
    st.markdown(
        "- **Budget Management:** Manage monthly media budgets upward of $1 million through Google, Meta, and Bing ad campaigns."
    )
    st.markdown(
        "- **Data Analysis:** Analyze performance trends and implement data-driven optimizations to lower CPA by over 25%."
    )
    st.markdown(
        "- **Experimentation:** Test targeting, bidding, and other data-driven strategies to increase conversion rate by over 40%."
    )
    
    st.markdown("---")
    
    # Experience 2
    st.markdown("### **Manager, Integrated Marketing & Analytics**")
    st.markdown("*The Madison Melle Agency* | *2022 - 2023*")
    st.markdown(
        "- **Campaign Management:** Built, launched, and owned all digital marketing efforts for 10+ clients across paid media channels."
    )
    st.markdown(
        "- **Revenue Strategy:** Influenced client revenue strategy, built pricing models, developed annual budget forecasts."
    )
    st.markdown(
        "- **Performance Analysis:** Compiled monthly performance reports with detailed analyses, presented findings to each client."
    )

    st.markdown("---")
    
    # Experience 3
    st.markdown("### **Strategic Campaign Manager**")
    st.markdown("*Top Floor Management* | *2021 - 2022*")
    st.markdown(
        "- **Strategic Marketing:** Generated strategies and tactics to successfully implement wide-scale social media campaigns."
    )
    st.markdown(
        "- **Campaign Execution:** Established partnerships, organized and executed brand marketing campaigns of 30+ content creators."
    )
    st.markdown(
        "- **Partnership Coordination:** Coordinated new and ongoing creator partnerships with brands, negotiating pricing and terms."
    )