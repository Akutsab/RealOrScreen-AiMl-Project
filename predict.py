import sys
from ultralytics import YOLO

# Load model
model = YOLO("best.pt")

# Image path from command line
image_path = sys.argv[1]

# Predict
result = model.predict(
    source=image_path,
    imgsz=224,
    verbose=False
)[0]

# Probability that image is a SCREEN
score = float(result.probs.data[1])

if(score>0.5):
  print("Screen:", score)
else:
  print("Real:", score)

