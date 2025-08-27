import streamlit as st
from rembg import remove
from PIL import Image
import numpy as np
from io import BytesIO
import base64
import os
import traceback
import time

import torch
from transformers import CLIPModel, CLIPProcessor

st.set_page_config(layout="wide", page_title="Image Background Remover + CLIP Zero-shot")

st.write("## Remove background + Zero-shot classification (CLIP)")
st.write(
    ":dog: 이미지를 업로드하면 배경을 제거하고, CLIP으로 입력한 라벨들과의 유사도 점수를 계산해요. "
    "rembg와 CLIP은 CPU에서도 동작합니다. 라벨을 쉼표로 구분해서 입력해보세요!"
)
st.sidebar.write("## Upload and options :gear:")

# -----------------------------
# Constants
# -----------------------------
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_IMAGE_SIZE = 2000  # pixels

# -----------------------------
# Helpers
# -----------------------------
def convert_image(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()

def resize_image(image: Image.Image, max_size: int) -> Image.Image:
    width, height = image.size
    if width <= max_size and height <= max_size:
        return image
    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))
    return image.resize((new_width, new_height), Image.LANCZOS)

def rgba_to_rgb(img_rgba: Image.Image, bg=(255, 255, 255)) -> Image.Image:
    """CLIP은 RGB 3채널을 기대. RGBA면 흰 배경으로 합성."""
    if img_rgba.mode == "RGBA":
        background = Image.new("RGB", img_rgba.size, bg)
        background.paste(img_rgba, mask=img_rgba.split()[-1])
        return background
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
    """rembg 처리 + 원본 반환 (캐시)"""
    try:
        image = Image.open(BytesIO(image_bytes))
        resized = resize_image(image, MAX_IMAGE_SIZE)
        fixed = remove(resized)  # RGBA
        return resized, fixed
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None

def clip_zero_shot(image_pil: Image.Image, labels, top_k: int = 5):
    """CLIP으로 라벨 유사도 계산."""
    model, processor = load_clip()
    image_rgb = rgba_to_rgb(image_pil)

    # labels 전처리
    labels = [l.strip() for l in labels if l.strip()]
    if not labels:
        return []

    with torch.no_grad():
        inputs = processor(
            text=labels,
            images=image_rgb,
            return_tensors="pt",
            padding=True
        )
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # [1, num_labels]
        probs = logits_per_image.softmax(dim=1)[0]   # [num_labels]

    scored = [{"label": t, "score": float(p)} for t, p in zip(labels, probs)]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:min(top_k, len(scored))]

def fix_image_and_classify(upload):
    try:
        start_time = time.time()
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()

        status_text.text("Loading image...")
        progress_bar.progress(10)

        # Read bytes
        if isinstance(upload, str):
            if not os.path.exists(upload):
                st.error(f"Default image not found at path: {upload}")
                return
            with open(upload, "rb") as f:
                image_bytes = f.read()
        else:
            image_bytes = upload.getvalue()

        # Sidebar: CLIP options
        st.sidebar.markdown("---")
        st.sidebar.write("### CLIP Options")
        default_labels = "t-shirt, jeans, jacket, sneakers, hat, dress, handbag, coat, skirt, sweatshirt"
        label_text = st.sidebar.text_area(
            "라벨 목록 (쉼표로 구분):",
            value=default_labels,
            height=80
        )
        top_k = st.sidebar.slider("Top-K", min_value=1, max_value=10, value=5, step=1)
        use_fixed_for_clip = st.sidebar.radio(
            "CLIP 입력 이미지 선택",
            options=("원본", "배경제거 결과"),
            index=1
        )

        status_text.text("Processing image (background removal)...")
        progress_bar.progress(35)

        # rembg 처리
        image, fixed = process_image(image_bytes)
        if image is None or fixed is None:
            return

        progress_bar.progress(70)
        status_text.text("Running CLIP...")

        # 어떤 이미지를 CLIP에 넣을지 선택
        clip_input_img = fixed if use_fixed_for_clip == "배경제거 결과" else image
        labels = [s for s in label_text.split(",")]
        clip_results = clip_zero_shot(clip_input_img, labels, top_k=top_k)

        progress_bar.progress(90)
        status_text.text("Displaying results...")

        # 2열 출력
        col1.write("Original Image :camera:")
        col1.image(image)

        col2.write("Background Removed :wrench:")
        col2.image(fixed)

        st.markdown("### CLIP 결과 (유사도 확률)")
        if clip_results:
            # 표로 출력
            st.dataframe(
                {
                    "label": [r["label"] for r in clip_results],
                    "score": [round(r["score"], 4) for r in clip_results],
                },
                use_container_width=True
            )
        else:
            st.info("라벨을 입력하면 CLIP 결과가 표시됩니다.")

        # 다운로드 버튼
        st.sidebar.download_button(
            "Download fixed image",
            convert_image(fixed),
            "fixed.png",
            "image/png"
        )

        progress_bar.progress(100)
        processing_time = time.time() - start_time
        status_text.text(f"Completed in {processing_time:.2f} seconds")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.sidebar.error("Failed to process image")
        print(f"Error in fix_image_and_classify: {traceback.format_exc()}")

# -----------------------------
# UI Layout
# -----------------------------
col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

with st.sidebar.expander("ℹ️ Image Guidelines"):
    st.write("""
    - Maximum file size: 10MB
    - Large images will be automatically resized to ≤ 2000px
    - Supported formats: PNG, JPG, JPEG
    - Processing time depends on image size
    """)

if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error(f"The uploaded file is too large. Please upload an image smaller than {MAX_FILE_SIZE/1024/1024:.1f}MB.")
    else:
        fix_image_and_classify(my_upload)
else:
    # 기본 샘플 이미지 시도
    default_images = ["./zebra.jpg", "./wallaby.png"]
    for img_path in default_images:
        if os.path.exists(img_path):
            fix_image_and_classify(img_path)
            break
    else:
        st.info("이미지를 업로드하면 배경 제거와 CLIP 분류 결과를 확인할 수 있어요!")