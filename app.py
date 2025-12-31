import streamlit as st
from pypdf import PdfWriter
import io

st.set_page_config(page_title="Unir PDFs", page_icon="📎")
st.title("📎 Unir documentos PDF")

uploaded_files = st.file_uploader(
    "Selecciona tus PDFs en el orden deseado",
    type=["pdf"],
    accept_multiple_files=True
)

output_name = st.text_input("Nombre del PDF final", "PDF_unido.pdf")

if uploaded_files and st.button("Unir PDFs"):
    writer = PdfWriter()

    for f in uploaded_files:
        writer.append(io.BytesIO(f.read()))

    merged_pdf = io.BytesIO()
    writer.write(merged_pdf)
    writer.close()
    merged_pdf.seek(0)

    st.success("PDF unido correctamente")
    st.download_button(
        "Descargar PDF",
        merged_pdf,
        file_name=output_name,
        mime="application/pdf"
    )
