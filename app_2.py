import streamlit as st
from PIL import Image
from yolo import YOLOLicensePlateDetector
from SR_GAN import SuperResolutionGAN
from paddleocr import PaddleOCR

# Set Streamlit page configuration
st.set_page_config(
    page_title="Ứng dụng xử lý biển số xe",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title of the app
st.markdown(
    """
    <h1 style="text-align:center; color:#FF4500;">
        🚗 ỨNG DỤNG LÀM RÕ NÉT BIỂN SỐ XE 🚗
    </h1>
    """,
    unsafe_allow_html=True
)

# Sidebar with instructions and confidence threshold slider
st.sidebar.header("Cài đặt:")
confidence_threshold = st.sidebar.slider(
    "Ngưỡng tin cậy YOLO (Confidence Threshold):", 
    0.1, 1.0, 0.5
)
st.sidebar.info(
    """
    - Tải lên một hình ảnh bằng cách nhấp vào nút "Choose an image".
    - Tùy chỉnh ngưỡng tin cậy YOLO để thay đổi mức độ chính xác của dự đoán.
    """
)

# Initialize models
yolo_model = YOLOLicensePlateDetector(confidence_threshold=confidence_threshold)
srgan = SuperResolutionGAN()
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# File uploader widget
uploaded_file = st.file_uploader("📂 Tải lên hình ảnh của bạn (JPG/PNG):", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open the uploaded image using PIL
    img = Image.open(uploaded_file)
    
    # Display the input image
    st.image(
        img, use_container_width=False, width=800, 
        caption="📷 Hình ảnh gốc"
    )
    
    # Perform YOLO detection
    st.markdown("### 🔍 Đang xử lý nhận diện biển số xe...")
    all_cropped_LP = yolo_model(img)
    
    # Process each cropped license plate with SRGAN
    st.markdown("### ✨ Đang làm rõ nét biển số...")
    sr_gan_results = [srgan(i) for i in all_cropped_LP]
    
    # Perform OCR on enhanced license plates
    st.markdown("### 📖 Đang nhận diện văn bản biển số...")
    result_ocr = [ocr.ocr(img, cls=True) for img in sr_gan_results]
    
    # Display results
    st.markdown("## 📋 KẾT QUẢ NHẬN DIỆN BIỂN SỐ:")
    
    # Loop through the list of cropped images
    for i, cropped_img in enumerate(all_cropped_LP):
        # Create columns for each step in the process
        col1, col2, col3 = st.columns([1, 1, 2])  # Adjust column width ratios as needed
        
        with col1:
            st.image(
                cropped_img, use_container_width=False, width=150, 
                caption="🖼️ Ảnh biển số cắt"
            )
        
        with col2:
            sr_img = Image.fromarray(sr_gan_results[i])
            st.image(
                sr_img, use_container_width=True, width=150, 
                caption="🔍 Ảnh làm rõ nét"
            )
            
        with col3:
            detected_text = [line[1][0] for line in result_ocr[i][0]]
            st.markdown(
                f"""
                <div style="padding:10px; background-color:#F0F8FF; border-radius:5px;">
                    <h4 style="color:#00008B;">Biển số nhận diện:</h4>
                    <p style="font-size:16px; color:#000;">{' '.join(detected_text)}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
else:
    st.markdown(
        """
        <h3 style="text-align:center; color:#808080;">
            Vui lòng tải lên một hình ảnh để bắt đầu 🖼️
        </h3>
        """,
        unsafe_allow_html=True
    )
