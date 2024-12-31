# from ultralytics import YOLO
# import numpy as np
# import os
# from PIL import Image
# import yaml
# import cv2
# from pathlib import Path
# class YOLOLicensePlateDetector:
#     task = "object_detection"
#     class_ = 2
#     def __init__(self):
#         #Load config
#         self.model = YOLO('best.pt')
#         self.confidence_threshold = 0.8
#         self.save_folder = 'out'
#         #warm up
#         dummy_input = np.random.randint(0, 256, (1, 640, 640, 3), dtype=np.uint8)
        
#     def process_image_path(self, img):
#         """Function to get all bounding box from an image_path
#         """
#         # Get the YOLO output
#         results = self.model([img])  
        
#         # Process YOLO output
#         all_cropped_LP = []
#         for _, result in enumerate(results):
#             # Lọc ra các bounding boxes có nhãn "license_plate" và conf > confidence_threshold
#             for idx, box, conf in zip(result.boxes.cls.cpu().numpy(), result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
#                 # Lọc những object thuộc class 1 - license_plate và conf > confidence_threshold
#                 if int(idx) == 1 and conf > self.confidence_threshold:  
#                     #crop bounding box
#                     x1, y1, x2, y2 = box
#                     cropped_plate = img.crop((x1, y1, x2, y2))
#                     cropped_plate_np = np.array(cropped_plate)
#                     # If needed, convert RGB to BGR (OpenCV format)
#                     # cropped_plate_np_bgr = cv2.cvtColor(cropped_plate_np, cv2.COLOR_RGB2BGR)
#                     all_cropped_LP.append(cropped_plate_np)
#                     print(cropped_plate)
#         return all_cropped_LP

#     def __call__(self,image_path):
#         return self.process_image_path(image_path)

# if __name__ == "__main__":
#     model = YOLOLicensePlateDetector()
#     image_path = "501.jpg"
#     all_cropped_LP = model(image_path)
#     print(all_cropped_LP)



from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2

class YOLOLicensePlateDetector:
    def __init__(self, model_path='best.pt', confidence_threshold=0.5):
        """
        Initializes the YOLO model with the specified confidence threshold.
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

    def process_image(self, img):
        """
        Processes an image using YOLO and returns cropped license plates.

        Args:
            img (PIL.Image or np.ndarray): Input image.

        Returns:
            list: List of cropped license plate images as numpy arrays.
        """
        # Ensure the image is a numpy array
        if isinstance(img, Image.Image):  # Convert PIL image to numpy array
            img = np.array(img)

        # Ensure image is in BGR format (OpenCV)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if img.shape[-1] == 3 else img

        # Run YOLO inference
        results = self.model.predict(img_bgr)

        # Process results
        all_cropped_LP = []
        for result in results:
            for idx, box, conf in zip(result.boxes.cls.cpu().numpy(),
                                      result.boxes.xyxy.cpu().numpy(),
                                      result.boxes.conf.cpu().numpy()):
                if int(idx) == 1 and conf >= self.confidence_threshold:  # Class 1: License Plate
                    x1, y1, x2, y2 = map(int, box)  # Convert to int
                    cropped_plate = img[y1:y2, x1:x2]  # Crop the license plate
                    all_cropped_LP.append(cropped_plate)
        
        return all_cropped_LP

    def __call__(self, image_input):
        """
        Detects license plates from an input image.

        Args:
            image_input (str or PIL.Image): Path to the image file or a PIL image.

        Returns:
            list: List of cropped license plate images as numpy arrays.
        """
        # Load image if image_input is a path
        if isinstance(image_input, str):
            img = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            img = image_input
        else:
            raise ValueError("Input must be a file path or a PIL.Image object.")
        
        return self.process_image(img)

if __name__ == "__main__":
    # Initialize YOLO detector
    model = YOLOLicensePlateDetector(model_path="best.pt", confidence_threshold=0.5)

    # Test with an image
    image_path = "501.jpg"  # Replace with your image path
    all_cropped_LP = model(image_path)

    # Display results
    for i, cropped_plate in enumerate(all_cropped_LP):
        cv2.imshow(f"Cropped Plate {i+1}", cropped_plate)
        cv2.waitKey(0)  # Wait for key press to close the window
    cv2.destroyAllWindows()
