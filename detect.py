from ultralytics import YOLO
import cv2
import os

def main():
    
    #Load the model
    model = YOLO("yolov8n.pt")

    #Input Data
    archive_dir = "archive"
    image_paths = [
        os.path.join(archive_dir, f) 
        for f in os.listdir(archive_dir) 
        if f.lower().endswith(('.jpg', '.png', '.jpeg'))
    ]

    #Detect objects in image
    results = model("archive/1.jpg")
    print(type(results))
    print("shape = ", len(results))
    os.makedirs("outputs", exist_ok=True)
    helper(model,image_paths)


    # Process results
    for result in results:
    #     # Option 1: Show detection (opens a window)
        result.show()  # Requires OpenCV

        # # Option 2: Save detection to file
        # result.save(filename="output.jpg")

        # # Option 3: Access bounding boxes manually (advanced)
        # boxes = result.boxes  # Boxes object
        # print(boxes.xyxy)

def helper(model,image_paths):
    for img_path in image_paths:
        results = model(img_path)
        output_name = f"detected_{os.path.basename(img_path)}"
        output_path = os.path.join("outputs", output_name)
        results[0].save(filename=output_path)
        print(f"Processed: {img_path} -> Saved to {output_path}")

if __name__ == "__main__":
    main()
