from ultralytics import YOLO
import cv2
import os

def main():
    
    #Load the model
    model = YOLO("yolov8n.pt")

    #Detect objects in image
    results = model("archive/1.jpg")
    print(type(results))
    print("shape = ", len(results))

    # Process results
    for result in results:
    #     # Option 1: Show detection (opens a window)
        result.show()  # Requires OpenCV

        # # Option 2: Save detection to file
        # result.save(filename="output.jpg")

        # # Option 3: Access bounding boxes manually (advanced)
        # boxes = result.boxes  # Boxes object
        # print(boxes.xyxy)



if __name__ == "__main__":
    main()
