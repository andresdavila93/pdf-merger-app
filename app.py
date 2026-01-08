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

APP_TITLE = "📄 Unir PDFs + ✍️ Firma opcional (firma arriba del nombre)"
TARGET_DEFAULT = "Lennin Karina Triana Fandiño"

# ----------------------------
# Session State init
# ----------------------------
def init_state():
    if "merged_pdf_bytes" not in st.session_state:
        st.session_state.merged_pdf_bytes = None

    if "detected" not in st.session_state:
        st.session_state.detected = False
    if "det_page" not in st.session_state:
        st.session_state.det_page = None
    if "det_rect" not in st.session_state:
        st.session_state.det_rect = None  # tuple (x0,y0,x1,y1)
    if "det_method" not in st.session_state:
        st.session_state.det_method = None

    if "last_output_name" not in st.session_state:
        st.session_state.last_output_name = "PDF_unido.pdf"
    if "last_target" not in st.session_state:
        st.session_state.last_target = TARGET_DEFAULT

init_state()

# ----------------------------
# Helpers
# ----------------------------
def normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def merge_pdfs_with_pymupdf(files_bytes_in_order) -> bytes:
    """Une PDFs en el orden dado, devuelve bytes."""
    merged = fitz.open()
    for pdf_bytes in files_bytes_in_order:
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
            return pi, rects[0]  # aparece una vez
    return None

def ocr_find_name_rect(doc: fitz.Document, target_text: str, zoom=2.8):
    """
    OCR fallback para PDFs escaneados.
    Retorna (page_index, rect_pdf) o None.
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

                # Convertir coords imagen(px) -> coords PDF
                page_rect = page.rect
                sx = page_rect.width / pix.width
                sy = page_rect.height / pix.height

                rect_pdf = fitz.Rect(x0 * sx, y0 * sy, x1 * sx, y1 * sy)
                return pi, rect_pdf

    return None

def render_page_image(page: fitz.Page, zoom=2.0) -> Image.Image:
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

def rect_pdf_to_img(rect_pdf: fitz.Rect, zoom: float):
    return (int(rect_pdf.x0 * zoom), int(rect_pdf.y0 * zoom),
            int(rect_pdf.x1 * zoom), int(rect_pdf.y1 * zoom))

def draw_highlight(img: Image.Image, rect_img, outline_width=6) -> Image.Image:
    """Rectángulo rojo para mostrar dónde está el nombre."""
    out = img.copy()
    d = ImageDraw.Draw(out)
    d.rectangle(rect_img, outline="red", width=outline_width)
    return out

def draw_signature_preview_above(img: Image.Image, rect_pdf: fitz.Rect, sig_img: Image.Image, zoom: float,
                                 gap=8, pad=6, scale_w=1.4, scale_h=2.0) -> Image.Image:
    """
    Preview: firma ARRIBA del nombre (no lo tapa).
    gap: espacio vertical entre firma y nombre (en px de la imagen renderizada).
    """
    out = img.copy()

    # Rect del nombre en imagen
    nx0, ny0, nx1, ny1 = rect_pdf_to_img(rect_pdf, zoom)
    name_w = max(1, nx1 - nx0)
    name_h = max(1, ny1 - ny0)

    # Tamaño firma
    fw = int(name_w * scale_w) + pad * 2
    fh = int(name_h * scale_h) + pad * 2

    # Alineación horizontal centrada
    cx = (nx0 + nx1) // 2
    fx0 = max(0, cx - fw // 2)
    fx1 = min(out.width, fx0 + fw)

    # Firma encima: su borde inferior queda arriba del nombre - gap
    fy1 = max(0, ny0 - gap)
    fy0 = max(0, fy1 - fh)

    # Evitar quedar pegado al borde superior
    if fy0 == 0 and (fy1 - fy0) < 10:
        fy1 = min(out.height, max(0, ny0 - 2))
        fy0 = max(0, fy1 - fh)

    sig = sig_img.convert("RGBA").resize((max(1, fx1 - fx0), max(1, fy1 - fy0)))
    out.paste(sig, (fx0, fy0), sig)
    return out

def insert_signature_above_into_pdf(doc: fitz.Document, page_index: int, name_rect: fitz.Rect, sig_bytes: bytes,
                                    gap=6, pad=4, scale_w=1.4, scale_h=2.0):
    """
    Inserta firma ARRIBA del nombre (no lo tapa).
    gap: espacio vertical entre firma y nombre (en puntos PDF).
    """
    page = doc[page_index]

    name_w = name_rect.x1 - name_rect.x0
    name_h = name_rect.y1 - name_rect.y0

    # Tamaño firma relativo al nombre
    w = name_w * scale_w + pad * 2
    h = name_h * scale_h + pad * 2

    # Centro X alineado al nombre (centrado)
    cx = (name_rect.x0 + name_rect.x1) / 2
    x0 = cx - w / 2
    x1 = cx + w / 2

    # Encima del nombre: borde inferior firma = arriba del nombre - gap
    y1 = name_rect.y0 - gap
    y0 = y1 - h

    # Evitar salir por arriba
    if y0 < 0:
        y0 = 0
        y1 = h

    rect_sig = fitz.Rect(x0, y0, x1, y1)
    page.insert_image(rect_sig, stream=sig_bytes, overlay=True)

def reset_all():
    st.session_state.merged_pdf_bytes = None
    st.session_state.detected = False
    st.session_state.det_page = None
    st.session_state.det_rect = None
    st.session_state.det_method = None

# ----------------------------
# UI
# ----------------------------
st.title(APP_TITLE)
st.write(
    "✅ **Esencia del app:** unir PDFs.\n\n"
    "Luego, si el PDF unido contiene el nombre indicado, puedes **firmar opcionalmente**.\n"
    "La firma se coloca **ARRIBA** del nombre (sin taparlo) y tienes **previsualización** antes de descargar."
)

colR1, colR2 = st.columns([1, 1])
with colR1:
    if st.button("🔄 Reiniciar", help="Borra el PDF unido actual y vuelve al paso 1"):
        reset_all()
with colR2:
    enable_ocr = st.toggle("Usar OCR si viene escaneado", value=True, key="enable_ocr")

uploaded_files = st.file_uploader(
    "📎 Paso 1: Sube tus PDFs (en el orden que quieres unirlos)",
    type=["pdf"],
    accept_multiple_files=True,
    key="pdfs_uploader"
)

target_name = st.text_input(
    "Nombre a detectar para habilitar firma (exacto)",
    value=st.session_state.last_target,
    key="target_name"
)
output_name = st.text_input(
    "Nombre del PDF final",
    value=st.session_state.last_output_name,
    key="output_name"
)

# Persistir inputs
st.session_state.last_target = target_name
st.session_state.last_output_name = output_name if output_name else "PDF_unido.pdf"

# Paso 1: Unir
if uploaded_files:
    st.write("**Archivos cargados:**")
    for i, f in enumerate(uploaded_files, start=1):
        st.write(f"{i}. {f.name}")

    if st.button("✅ Unir PDFs", type="primary", key="merge_btn"):
        files_bytes = [f.getvalue() for f in uploaded_files]
        merged_bytes = merge_pdfs_with_pymupdf(files_bytes)

        st.session_state.merged_pdf_bytes = merged_bytes

        # Detectar nombre en el PDF unido
        doc = fitz.open(stream=merged_bytes, filetype="pdf")

        found = find_name_rect_text(doc, target_name)
        method = "texto"

        if not found and enable_ocr:
            method = "ocr"
            found_ocr = ocr_find_name_rect(doc, target_name, zoom=2.8)
            if found_ocr:
                found = found_ocr

        if found:
            page_index, rect = found
            st.session_state.detected = True
            st.session_state.det_page = int(page_index)
            st.session_state.det_rect = (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))
            st.session_state.det_method = method
        else:
            st.session_state.detected = False
            st.session_state.det_page = None
            st.session_state.det_rect = None
            st.session_state.det_method = None

        doc.close()
        st.success("PDFs unidos ✅ (resultado guardado). Baja al Paso 2 para descargar o firmar.")

# Paso 2: Descargar / Firmar
if st.session_state.merged_pdf_bytes:
    st.divider()
    st.header("Paso 2: Descargar (y firmar opcionalmente)")

    merged_pdf_bytes = st.session_state.merged_pdf_bytes

    st.download_button(
        "⬇️ Descargar PDF unido (sin firma)",
        data=merged_pdf_bytes,
        file_name=output_name if output_name.lower().endswith(".pdf") else output_name + ".pdf",
        mime="application/pdf",
        key="dl_merged"
    )

    if not st.session_state.detected:
        st.info("No se detectó el nombre en el PDF unido, por eso no se habilita la firma.")
    else:
        st.success(
            f"Nombre detectado ✅ Método: **{st.session_state.det_method}**. "
            f"Página: **{st.session_state.det_page + 1}**"
        )

        preview_zoom = st.slider("Zoom de previsualización", 1.0, 3.5, 2.0, 0.1, key="preview_zoom")

        # Abrir doc desde bytes para preview
        doc = fitz.open(stream=merged_pdf_bytes, filetype="pdf")
        page = doc[st.session_state.det_page]
        rect_pdf = fitz.Rect(*st.session_state.det_rect)

        img_page = render_page_image(page, zoom=preview_zoom)
        rect_img = rect_pdf_to_img(rect_pdf, zoom=preview_zoom)

        st.subheader("👀 Previsualización (donde se encontró el nombre)")
        st.image(draw_highlight(img_page, rect_img), use_column_width=True)

        wants_sign = st.toggle("✍️ ¿Deseas firmar este documento?", value=False, key="wants_sign")

        if wants_sign:
            sig_file = st.file_uploader("Sube la firma (PNG/JPG)", type=["png", "jpg", "jpeg"], key="sig_uploader")

            # Controles firma
            gap = st.slider("Espacio entre firma y nombre (gap)", 0, 40, 10, key="gap")
            pad = st.slider("Margen (padding)", 0, 20, 6, key="pad")
            scale_w = st.slider("Escala ancho firma", 0.8, 2.5, 1.4, 0.1, key="scale_w")
            scale_h = st.slider("Escala alto firma", 0.8, 4.0, 2.0, 0.1, key="scale_h")

            if sig_file:
                sig_img = Image.open(sig_file).convert("RGBA")

                st.subheader("✅ Preview con firma (firma arriba del nombre)")
                st.image(
                    draw_signature_preview_above(
                        img_page, rect_pdf, sig_img,
                        zoom=preview_zoom,
                        gap=gap, pad=pad, scale_w=scale_w, scale_h=scale_h
                    ),
                    use_column_width=True
                )

                if st.button("🔒 Confirmar y generar PDF firmado", type="primary", key="confirm_sign"):
                    doc2 = fitz.open(stream=merged_pdf_bytes, filetype="pdf")
                    rect_pdf2 = fitz.Rect(*st.session_state.det_rect)

                    insert_signature_above_into_pdf(
                        doc2,
                        st.session_state.det_page,
                        rect_pdf2,
                        sig_file.getvalue(),
                        gap=gap, pad=pad, scale_w=scale_w, scale_h=scale_h
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
                        mime="application/pdf",
                        key="dl_signed"
                    )
            else:
                st.warning("Sube la imagen de firma para previsualizar y confirmar.")
        doc.close()
else:
    st.info("Sube PDFs y haz clic en **Unir PDFs** para continuar.")
