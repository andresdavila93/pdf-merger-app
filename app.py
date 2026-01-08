import io
import re
import streamlit as st
import fitz  # PyMuPDF
from PIL import Image, ImageDraw
import pytesseract


# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="Unir PDFs + Firma opcional", page_icon="📄", layout="centered")

TARGET_DEFAULT = "Lennin Karina Triana Fandiño"  # puedes cambiarlo
APP_TITLE = "📄 Unir PDFs + ✍️ Firma opcional sobre nombre"


# ----------------------------
# Helpers
# ----------------------------
def normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def merge_pdfs_with_pymupdf(uploaded_files) -> bytes:
    """Une PDFs en el orden cargado, devuelve bytes del PDF unido."""
    merged = fitz.open()
    for f in uploaded_files:
        pdf_bytes = f.read()
        src = fitz.open(stream=pdf_bytes, filetype="pdf")
        merged.insert_pdf(src)
        src.close()

    out = io.BytesIO()
    merged.save(out)
    merged.close()
    out.seek(0)
    return out.getvalue()


def find_name_rect_text(doc: fitz.Document, target_text: str):
    """Busca texto seleccionable. Retorna (page_index, rect) o None."""
    for pi in range(doc.page_count):
        page = doc[pi]
        rects = page.search_for(target_text)
        if rects:
            return pi, rects[0]  # aparece una vez (según tu caso)
    return None


def ocr_find_name_rect(doc: fitz.Document, target_text: str, zoom=2.8):
    """
    OCR fallback para PDFs escaneados.
    Retorna (page_index, rect_pdf, zoom_used) o None.
    """
    target_tokens = normalize(target_text).split()

    for pi in range(doc.page_count):
        page = doc[pi]

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
            words.append((txt, x, y, x + w, y + h))

        # Buscar secuencia exacta de tokens
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

                # Convertir coords de imagen(px) -> coords PDF
                page_rect = page.rect
                sx = page_rect.width / pix.width
                sy = page_rect.height / pix.height

                rect_pdf = fitz.Rect(x0 * sx, y0 * sy, x1 * sx, y1 * sy)
                return pi, rect_pdf, zoom

    return None


def render_page_image(page: fitz.Page, zoom=2.0) -> Image.Image:
    """Renderiza una página del PDF a imagen para previsualización."""
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img


def rect_pdf_to_img(rect_pdf: fitz.Rect, zoom: float) -> tuple[int, int, int, int]:
    """Convierte rect en coords PDF a coords en imagen renderizada con zoom."""
    return (int(rect_pdf.x0 * zoom), int(rect_pdf.y0 * zoom),
            int(rect_pdf.x1 * zoom), int(rect_pdf.y1 * zoom))


def draw_highlight(img: Image.Image, rect_img, outline_width=6) -> Image.Image:
    """Dibuja un rectángulo para que el usuario vea dónde se detectó el nombre."""
    out = img.copy()
    d = ImageDraw.Draw(out)
    x0, y0, x1, y1 = rect_img
    # marco (sin color específico no es posible; usamos el default de PIL que es visible)
    d.rectangle([x0, y0, x1, y1], outline="red", width=outline_width)
    return out


def draw_signature_preview(img: Image.Image, rect_pdf: fitz.Rect, sig_img: Image.Image, zoom: float,
                           pad=6, scale_w=1.4, scale_h=1.8) -> Image.Image:
    """
    Previsualiza firma sobre la IMAGEN (no toca el PDF).
    Calcula un rect de firma centrado en el rect del nombre.
    """
    out = img.copy()

    # Rect del nombre en imagen
    nx0, ny0, nx1, ny1 = rect_pdf_to_img(rect_pdf, zoom)
    name_w = max(1, nx1 - nx0)
    name_h = max(1, ny1 - ny0)

    # Rect firma (en imagen), centrado
    fw = int(name_w * scale_w) + pad * 2
    fh = int(name_h * scale_h) + pad * 2
    cx = (nx0 + nx1) // 2
    cy = (ny0 + ny1) // 2
    fx0 = max(0, cx - fw // 2)
    fy0 = max(0, cy - fh // 2)
    fx1 = min(out.width, fx0 + fw)
    fy1 = min(out.height, fy0 + fh)

    sig_resized = sig_img.resize((max(1, fx1 - fx0), max(1, fy1 - fy0)))

    # Pegar con alpha si existe
    if sig_resized.mode != "RGBA":
        sig_resized = sig_resized.convert("RGBA")
    out.paste(sig_resized, (fx0, fy0), sig_resized)

    return out


def insert_signature_into_pdf(doc: fitz.Document, page_index: int, name_rect: fitz.Rect, sig_bytes: bytes,
                              pad=6, scale_w=1.4, scale_h=1.8):
    """Inserta firma (imagen) sobre el nombre en el PDF real."""
    page = doc[page_index]

    # Rect firma en coords PDF centrado en el rect del nombre
    name_w = name_rect.x1 - name_rect.x0
    name_h = name_rect.y1 - name_rect.y0
    w = name_w * scale_w
    h = name_h * scale_h
    cx = (name_rect.x0 + name_rect.x1) / 2
    cy = (name_rect.y0 + name_rect.y1) / 2

    rect_sig = fitz.Rect(
        cx - w / 2 - pad, cy - h / 2 - pad,
        cx + w / 2 + pad, cy + h / 2 + pad
    )

    page.insert_image(rect_sig, stream=sig_bytes, overlay=True)


# ----------------------------
# UI
# ----------------------------
st.title(APP_TITLE)
st.write("1) Une tus PDFs. 2) Si el PDF unido contiene el nombre, puedes **firmar opcionalmente** encima (con preview).")

uploaded_files = st.file_uploader(
    "📎 Sube tus PDFs (en el orden que quieres unirlos)",
    type=["pdf"],
    accept_multiple_files=True
)

target_name = st.text_input("Nombre a detectar para habilitar firma (exacto)", value=TARGET_DEFAULT)

colA, colB = st.columns(2)
with colA:
    output_name = st.text_input("Nombre del PDF final", value="PDF_unido.pdf")
with colB:
    enable_ocr = st.toggle("Usar OCR si viene escaneado (recomendado)", value=True)

if uploaded_files and len(uploaded_files) >= 1:
    st.write("**Archivos cargados:**")
    for i, f in enumerate(uploaded_files, start=1):
        st.write(f"{i}. {f.name}")

    if st.button("✅ Unir PDFs", type="primary"):
        # Unir
        merged_pdf_bytes = merge_pdfs_with_pymupdf(uploaded_files)
        st.success("PDFs unidos correctamente ✅")

        # Abrir el unido para detectar nombre
        doc = fitz.open(stream=merged_pdf_bytes, filetype="pdf")

        found = find_name_rect_text(doc, target_name)
        found_method = "texto"
        ocr_zoom_used = None

        if not found and enable_ocr:
            found_method = "ocr"
            found_ocr = ocr_find_name_rect(doc, target_name, zoom=2.8)
            if found_ocr:
                page_index, rect_pdf, ocr_zoom_used = found_ocr
                found = (page_index, rect_pdf)

        # Descarga “sin firma” siempre disponible
        st.download_button(
            "⬇️ Descargar PDF unido (sin firma)",
            data=merged_pdf_bytes,
            file_name=output_name if output_name.lower().endswith(".pdf") else output_name + ".pdf",
            mime="application/pdf"
        )

        if not found:
            st.info("No se detectó el nombre en el PDF unido. No se habilita firma.")
            doc.close()
        else:
            page_index, rect_pdf = found
            st.success(f"Nombre detectado ✅ Método: **{found_method}**. Página: **{page_index + 1}**")

            # Preview de la página detectada
            preview_zoom = st.slider("Zoom de previsualización", 1.0, 3.5, 2.0, 0.1)
            page = doc[page_index]
            img_page = render_page_image(page, zoom=preview_zoom)
            rect_img = rect_pdf_to_img(rect_pdf, zoom=preview_zoom)

            st.subheader("👀 Previsualización (donde se encontró el nombre)")
            st.image(draw_highlight(img_page, rect_img), use_column_width=True)

            # Firma opcional
            wants_sign = st.toggle("✍️ ¿Deseas firmar este documento?", value=False)

            if not wants_sign:
                st.info("Firma desactivada. Si activas el switch podrás subir la firma y ver preview antes de descargar.")
                doc.close()
            else:
                sig_file = st.file_uploader("Sube la firma (PNG/JPG)", type=["png", "jpg", "jpeg"])

                pad = st.slider("Margen (padding) alrededor", 0, 20, 6)
                scale_w = st.slider("Escala ancho firma", 0.8, 2.5, 1.4, 0.1)
                scale_h = st.slider("Escala alto firma", 0.8, 3.5, 1.8, 0.1)

                if sig_file:
                    sig_img = Image.open(sig_file).convert("RGBA")

                    st.subheader("✅ Preview con firma (solo visual)")
                    st.image(
                        draw_signature_preview(img_page, rect_pdf, sig_img, zoom=preview_zoom,
                                               pad=pad, scale_w=scale_w, scale_h=scale_h),
                        use_column_width=True
                    )

                    if st.button("🔒 Confirmar y generar PDF firmado", type="primary"):
                        # Re-abrimos doc desde bytes originales unidos (para no depender del estado actual)
                        doc2 = fitz.open(stream=merged_pdf_bytes, filetype="pdf")

                        # Re-detectar rect en doc2 (mismo índice). En la práctica, el doc es el mismo.
                        # Si fue OCR, el rect_pdf ya está en coords PDF y aplica igual.
                        sig_bytes = sig_file.getvalue()

                        insert_signature_into_pdf(
                            doc2, page_index, rect_pdf, sig_bytes,
                            pad=pad, scale_w=scale_w, scale_h=scale_h
                        )

                        out = io.BytesIO()
                        doc2.save(out)
                        doc2.close()
                        out.seek(0)

                        st.success("PDF firmado generado ✅")
                        st.download_button(
                            "⬇️ Descargar PDF unido y firmado",
                            data=out,
                            file_name="PDF_unido_firmado.pdf",
                            mime="application/pdf"
                        )

                else:
                    st.warning("Sube la imagen de la firma para ver la previsualización y firmar.")

else:
    st.info("Sube al menos 1 PDF para comenzar (ideal 2 o más).")
