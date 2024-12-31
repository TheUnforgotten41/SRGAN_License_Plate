import streamlit as st

st.set_page_config(
    page_title="Thesis_defense",
    page_icon="huet_logo_2.png",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)
from PIL import Image
from yolo import YOLOLicensePlateDetector
from SR_GAN import SuperResolutionGAN

# Title of the app
st.title(":red_car: ỨNG DỤNG LÀM RÕ NÉT BIỂN SỐ XE")

yolo_model = YOLOLicensePlateDetector()
srgan = SuperResolutionGAN()
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image

# Initialize OCR model
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])



# result = ocr.ocr(image_np, cls=True)
if uploaded_file is not None:
    # Open the uploaded image using PIL
    img = Image.open(uploaded_file)
    
    st.image(img, use_container_width=True)
    
    # Display the image

    all_cropped_LP = yolo_model(img)
    sr_gan_results = [srgan(i) for i in all_cropped_LP]
    result_ocr =[ocr.ocr(img, cls=True)   for img in sr_gan_results]
    
    # Title of the app
    st.title("Biển số nhận diện được:")
    # Loop through the list of images
    for i,img in enumerate(all_cropped_LP):
        # Open the image using PIL
        # img = Image.open(image_path)
        
        # Create two columns: one for the image, one for the button
        col1, col2, col3 = st.columns([1, 1,1])  # Adjust the column width as needed
        
        with col1:
            # Display the image
            st.image(img, use_container_width=False, width=100)
        
        with col2:
            sr_img = Image.fromarray(sr_gan_results[i])
            st.image(sr_img, use_container_width=True)
            
        with col3:
            st.write( [line[1][0] for line in result_ocr[i][0]])
