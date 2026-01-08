import io, re
import streamlit as st
import fitz  # pymupdf
from PIL import Image
import pytesseract

st.set_page_config(page_title="Unir y firmar PDF", page_icon="✍️")
st.title("✍️ Firma opcional sobre el nombre (con OCR si está escaneado)")

TARGET_DEFAULT = "Lennin Karina Triana Fandiño"

def normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def find_text_rect(doc, target_text: str):
    """Busca texto real seleccionable. Devuelve (page_index, rect) o None."""
    for pi, page in enumerate(doc):
        rects = page.search_for(target_text)
        if rects:
            return pi, rects[0]  # aparece una vez
    return None

def ocr_find_rect(doc, target_text: str):
    """OCR para PDFs escaneados. Devuelve (page_index, rect_en_coords_pdf) o None."""
    target_tokens = normalize(target_text).split()

    for pi, page in enumerate(doc):
        zoom = 2.8
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

        words = []
        for i in range(len(data["text"])):
            txt = normalize(data["text"][i])
            if not txt:
                continue
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            words.append((txt, x, y, x+w, y+h))

        for start in range(0, len(words) - len(target_tokens) + 1):
            ok = True
            for j, tok in enumerate(target_tokens):
                if words[start + j][0] != tok:
                    ok = False
                    break
            if ok:
                x0 = min(words[start + j][1] for j in range(len(target_tokens)))
                y0 = min(words[start + j][2] for j in range(len(target_tokens)))
                x1 = max(words[start + j][3] for j in range(len(target_tokens)))
                y1 = max(words[start + j][4] for j in range(len(target_tokens)))

                # Convertir px → coords PDF
                page_rect = page.rect
                sx = page_rect.width / pix.width
                sy = page_rect.height / pix.height

                rect_pdf = fitz.Rect(x0*sx, y0*sy, x1*sx, y1*sy)
                return pi, rect_pdf

    return None

def insert_signature(doc, page_index, name_rect, sig_bytes, pad=6, scale_w=1.4, scale_h=1.8):
    page = doc[page_index]

    # Rect para firma, centrada sobre el rect del nombre
    w = (name_rect.x1 - name_rect.x0) * scale_w
    h = (name_rect.y1 - name_rect.y0) * scale_h
    cx = (name_rect.x0 + name_rect.x1) / 2
    cy = (name_rect.y0 + name_rect.y1) / 2

    rect_sig = fitz.Rect(
        cx - w/2 - pad, cy - h/2 - pad,
        cx + w/2 + pad, cy + h/2 + pad
    )

    page.insert_image(rect_sig, stream=sig_bytes, overlay=True)

# UI
target_name = st.text_input("Nombre a detectar", value=TARGET_DEFAULT)
pdf_file = st.file_uploader("Sube el PDF", type=["pdf"])

if pdf_file:
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")

    # 1) Detectar por texto real
    found = find_text_rect(doc, target_name)
    method = "texto"
    # 2) Si no, OCR
    if not found:
        method = "ocr"
        found = ocr_find_rect(doc, target_name)

    if not found:
        st.error("No encontré el nombre. Puede estar escrito diferente o el escaneo está muy borroso.")
        doc.close()
    else:
        page_index, rect = found
        st.success(f"Nombre detectado ✅ (método: {method}). Página: {page_index + 1}")

        # Opción de firma (del usuario)
        wants_sign = st.toggle("¿Deseas firmar este documento?", value=False)

        if wants_sign:
            sig_file = st.file_uploader("Sube la firma (PNG/JPG)", type=["png", "jpg", "jpeg"])
            pad = st.slider("Margen", 0, 20, 6)
            scale_w = st.slider("Escala ancho", 0.8, 2.5, 1.4, 0.1)
            scale_h = st.slider("Escala alto", 0.8, 3.5, 1.8, 0.1)

            if sig_file and st.button("Firmar ahora", type="primary"):
                sig_bytes = sig_file.read()
                insert_signature(doc, page_index, rect, sig_bytes, pad=pad, scale_w=scale_w, scale_h=scale_h)

                out = io.BytesIO()
                doc.save(out)
                doc.close()
                out.seek(0)

                st.download_button(
                    "⬇️ Descargar PDF firmado",
                    data=out,
                    file_name="pdf_firmado.pdf",
                    mime="application/pdf"
                )
        else:
            st.info("Firma desactivada. Puedes activar el switch si deseas firmar.")
            doc.close()
