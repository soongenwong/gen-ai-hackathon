import streamlit as st
import pandas as pd
from backend import load_dataframe, generate_synthetic_data
from fpdf import FPDF
import tempfile

def dataframe_to_pdf_bytes(df, max_rows=200):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=8)
    col_width = pdf.w / (len(df.columns) + 1)
    row_height = pdf.font_size * 1.5

    # Header
    for col in df.columns:
        pdf.cell(col_width, row_height, str(col), border=1)
    pdf.ln(row_height)

    # Rows (limit to max_rows for performance)
    for i, row in df.head(max_rows).iterrows():
        for item in row:
            pdf.cell(col_width, row_height, str(item), border=1)
        pdf.ln(row_height)

    # Save to bytes
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        pdf.output(tmp.name)
        tmp.seek(0)
        pdf_bytes = tmp.read()
    return pdf_bytes

# ──────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Synthetic Data Generator",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────────────────────────────────────
# Main App Structure
# ──────────────────────────────────────────────────────────────────────────────
def main():
    # Header
    st.title("🔬 Synthetic Data Generator")
    st.markdown("Generate high-quality synthetic data using advanced machine learning models")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        st.info("Upload your data and configure settings here")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Data Input")
        st.markdown("Upload your data file to get started (CSV, Excel, or other supported formats)")
        
        uploaded_file = st.file_uploader("Upload File", type=None)  # Accept any file type
        if uploaded_file is not None:
            try:
                df = load_dataframe(uploaded_file)
                st.session_state['df'] = df  # Store in session state
                st.success("File loaded successfully!")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"Error loading file: {e}")
        elif 'df' in st.session_state:
            df = st.session_state['df']
            st.info("Using previously uploaded file.")
            st.dataframe(df.head())
        else:
            df = None
            st.info("Please upload a data file.")
        
    with col2:
        st.header("Settings")
        st.markdown("Configure your synthetic data generation")
        n_rows = st.number_input("Number of synthetic rows", min_value=1, value=1000)
        model_type = st.selectbox("Model type", ["ctgan", "copulagan", "tvae", "gaussiancopula"])
        privacy_level = st.slider("Privacy level", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        generate_btn = st.button("Generate Synthetic Data")
    
    # Results section
    st.header("Results")
    if generate_btn:
        if 'df' in st.session_state and st.session_state['df'] is not None:
            try:
                synth_df = generate_synthetic_data(
                    st.session_state['df'],
                    n_rows=int(n_rows),
                    model_type=model_type,
                    privacy_level=privacy_level
                )
                st.success(f"Generated {len(synth_df)} synthetic rows.")
                st.dataframe(synth_df.head())
                csv = synth_df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Synthetic Data as CSV", csv, "synthetic_data.csv", "text/csv")
                pdf_bytes = dataframe_to_pdf_bytes(synth_df, max_rows=200)
                st.download_button(
                    "Download Synthetic Data as PDF",
                    pdf_bytes,
                    "synthetic_data.pdf",
                    "application/pdf"
                )
            except Exception as e:
                st.error(f"Error generating synthetic data: {e}")
        else:
            st.warning("Please upload a valid data file before generating synthetic data.")
    else:
        st.info("Generated synthetic data will appear here")
    
    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit and SDV")

if __name__ == "__main__":
    main()