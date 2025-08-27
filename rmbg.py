import streamlit as st
from rembg import remove
from PIL import Image
from io import BytesIO
import os, traceback, time
import torch
from transformers import CLIPModel, CLIPProcessor

st.set_page_config(layout="wide", page_title="Image Background Remover + CLIP Zero-shot")
st.write("## Remove background + Zero-shot classification (CLIP)")
st.sidebar.write("## Upload and options :gear:")

# -----------------------------
# 고정 카테고리 목록 (여기만 수정해서 쓰세요)
# -----------------------------
category = [
"t-shirt", "short sleeve shirt or blouse", "long sleeve shirt or blouse", "knitwear or sweater", "sweatshirt", "short sleeve t-shirt", "tank top",
"denim pants or jeans", "half pants", "jogger pants", "cotton pants", "slacks", "leggings",
"mini skirt", "midi skirt", "long skirt", "onepiece dress",
"short padding", "sheepskin jacket", "zip-up hoodie", "windbreak", "leather jacket", "denim jacket", "blazer", "cardigan", "anorak", "fleece", "coat", "long padding", "padding vest",
"sneakers", "boots", "dress shoes", "sandal or slipper",
"crossbody bag", "shoulder bag", "backpack", "tote bog", "eco bag",
"hat", "scarf", "socks", "wristwatch", "ring or neckless or jewerly", "belt", "glasses"
]

# -----------------------------
# Constants
# -----------------------------
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_IMAGE_SIZE = 2000  # px

# -----------------------------
# Helpers
# -----------------------------
def convert_image(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()

def resize_image(image: Image.Image, max_size: int) -> Image.Image:
    w, h = image.size
    if w <= max_size and h <= max_size:
        return image
    if w > h:
        nw, nh = max_size, int(h * (max_size / w))
    else:
        nh, nw = max_size, int(w * (max_size / h))
    return image.resize((nw, nh), Image.LANCZOS)

def rgba_to_rgb(img_rgba: Image.Image, bg=(255, 255, 255)) -> Image.Image:
    if img_rgba.mode == "RGBA":
        bg_img = Image.new("RGB", img_rgba.size, bg)
        bg_img.paste(img_rgba, mask=img_rgba.split()[-1])
        return bg_img
    return img_rgba.convert("RGB")

@st.cache_resource(show_spinner=False)
def load_clip():
    MODEL_ID = "laion/CLIP-ViT-B-32-laion2B-s34B-b79k"
    model = CLIPModel.from_pretrained(MODEL_ID)
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    model.eval()
    return model, processor

@st.cache_data(show_spinner=False)
def process_image(image_bytes: bytes):
    try:
        img = Image.open(BytesIO(image_bytes))
        resized = resize_image(img, MAX_IMAGE_SIZE)
        fixed = remove(resized)  # RGBA
        return resized, fixed
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, None

def clip_scores(image_pil: Image.Image, labels):
    """labels는 고정된 `category` 리스트 사용"""
    if not labels:
        return []
    model, processor = load_clip()
    image_rgb = rgba_to_rgb(image_pil)
    with torch.no_grad():
        inputs = processor(text=labels, images=image_rgb, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)[0]  # [num_labels]
    scored = [{"label": t, "score": float(p)} for t, p in zip(labels, probs)]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored

def run_pipeline(upload):
    try:
        start = time.time()
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()

        status_text.text("Loading image...")
        progress_bar.progress(10)

        # Read bytes
        if isinstance(upload, str):
            if not os.path.exists(upload):
                st.error(f"Default image not found at {upload}")
                return
            with open(upload, "rb") as f:
                image_bytes = f.read()
        else:
            image_bytes = upload.getvalue()

        st.sidebar.markdown("---")
        st.sidebar.write("### CLIP Options")
        use_fixed_for_clip = st.sidebar.radio(
            "CLIP 입력 이미지",
            options=("원본", "배경제거 결과"),
            index=1
        )

        status_text.text("Processing image (background removal)...")
        progress_bar.progress(35)

        # rembg
        image, fixed = process_image(image_bytes)
        if image is None or fixed is None:
            return

        progress_bar.progress(70)
        status_text.text("Running CLIP...")

        clip_input = fixed if use_fixed_for_clip == "배경제거 결과" else image
        scored = clip_scores(clip_input, category)

        progress_bar.progress(90)
        status_text.text("Displaying results...")

        # UI 2 columns
        col1, col2 = st.columns(2)
        with col1:
            st.write("Original Image :camera:")
            st.image(image)
        with col2:
            st.write("Background Removed :wrench:")
            st.image(fixed)

        st.markdown("### CLIP 결과 (유사도 확률)")
        if scored:
            # 드롭다운: 옵션은 category 전체, 기본값은 Top-1
            top1_label = scored[0]["label"]
            try:
                default_index = category.index(top1_label)
            except ValueError:
                default_index = 0

            selected_label = st.selectbox(
                "예측 결과 (수정 가능)",
                options=category,
                index=default_index,
                key="predicted_category"
            )

            # 표 표시
            st.dataframe(
                {
                    "label": [r["label"] for r in scored],
                    "score": [round(r["score"], 4) for r in scored],
                },
                use_container_width=True
            )

            st.info(f"드롭다운 기본값은 Top-1 예측: **{top1_label}** 입니다. (선택: **{selected_label}**)")

        else:
            st.warning("카테고리 목록이 비어 있습니다. `category` 리스트를 확인하세요.")

        # Download button
        st.sidebar.download_button(
            "Download fixed image",
            convert_image(fixed),
            "fixed.png",
            "image/png"
        )

        progress_bar.progress(100)
        status_text.text(f"Completed in {time.time() - start:.2f} seconds")

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.sidebar.error("Failed to process image")
        print("TRACE:", traceback.format_exc())

# -----------------------------
# UI
# -----------------------------
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

with st.sidebar.expander("ℹ️ Image Guidelines"):
    st.write("""
    - Maximum file size: 10MB
    - Large images auto-resized to ≤ 2000px
    - Formats: PNG, JPG, JPEG
    - Processing time depends on size
    """)
with st.sidebar.expander("📦 Categories (fixed)"):
    st.write(", ".join(category))

if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error(f"The uploaded file is too large. Please upload an image smaller than {MAX_FILE_SIZE/1024/1024:.1f}MB.")
    else:
        run_pipeline(my_upload)
else:
    defaults = ["./zebra.jpg", "./wallaby.png"]
    for p in defaults:
        if os.path.exists(p):
            run_pipeline(p)
            break
    else:
        st.info("이미지를 업로드하면 배경 제거와 CLIP 예측 결과(드롭다운 기본값=Top-1)가 표시됩니다!")