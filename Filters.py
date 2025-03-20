import cv2
import streamlit as st
import numpy as np
from PIL import Image

st.title("Filters App")


def blacknwhite(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return gray_img


def pencil_sketch(img, ksize=5):
    blur = cv2.GaussianBlur(img, (ksize, ksize), 0, 0)
    sketch, _ = cv2.pencilSketch(blur)
    return sketch


def HDR(img, sigma_s=10, sigma_r=0.1):
    hd_img = cv2.detailEnhance(img, sigma_s=sigma_s, sigma_r=sigma_r)
    return hd_img


def stylezation(img, sigma_s=10, sigma_r=0.1):
    blur = cv2.GaussianBlur(img, (5, 5), 0, 0)
    style = cv2.stylization(blur, sigma_s=sigma_s, sigma_r=sigma_r)
    return style


def brightness(img, level):
    bright = cv2.convertScaleAbs(img, beta=level)
    return bright


upload = st.file_uploader("Please Upload an image", type=["png", "jpeg", "jpg", "webp"])

if upload is not None:
    img = Image.open(upload)

    # convert from PIl to cv2 using numpy array
    img_array = np.array(img)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

    original_image, filtered_image = st.columns(2)
    with original_image:
        st.header("Original Image")
        st.image(img_array, channels="BGR", use_column_width=True)

    options = st.selectbox("Select filter", ("None", "Black&White", "pencilsketch", "HDR", "stylezation", "brightness"))
    output_flag = 1
    color = "BGR"
    if options == "None":
        output_flag = 0
        output = img_array
    elif options == "Black&White":
        output = blacknwhite(img_array)
        color = "GRAY"
    elif options == "pencilsketch":
        kvalue = st.slider("Kernel size", 1, 9, 4)
        output = pencil_sketch(img_array, kvalue)
        color = "GRAY"
    elif options == "HDR":
        image = brightness(img_array, 30)
        output = HDR(image)
    elif options == "stylezation":
        sigma_r = st.slider("sigma_r", 0, 10, 2)
        sigma_s = st.slider("sigma_s", 0, 1, 2)
        output = stylezation(img_array, sigma_r, sigma_s)
    elif options == "brightness":
        level = st.slider("level", -50, 50, 2)
        output = brightness(img_array, level)
    with filtered_image:
        st.header("Filtered Image")
        st.image(output, channels=color, use_column_width=True)