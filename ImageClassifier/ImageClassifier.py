# Import required libraries
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# -----------------------------
# 1. Load the pretrained model
# -----------------------------
model = VGG16(weights='imagenet')
print("VGG16 model loaded successfully!")

# -----------------------------
# 2. Load and prepare the image
# -----------------------------
# CHANGE THIS to your actual file path
img_path = "/Users/yourname/Desktop/dog.jpg"

# Load and resize image to 224x224
img = image.load_img(img_path, target_size=(224, 224))

# Display the image
plt.imshow(img)
plt.axis('off')
plt.title("Input Image")
plt.show()

# Convert image to NumPy array
img_array = image.img_to_array(img)

# Expand dimensions to create batch (1, 224, 224, 3)
img_array = np.expand_dims(img_array, axis=0)

# Preprocess the image for VGG16
img_array = preprocess_input(img_array)

# -----------------------------
# 3. Make predictions
# -----------------------------
predictions = model.predict(img_array)

# -----------------------------
# 4. Decode top 5 predictions
# -----------------------------
decoded = decode_predictions(predictions, top=5)[0]

print("\nTop 5 Predictions:")
for i, (imagenet_id, label, confidence) in enumerate(decoded):
    print(f"{i+1}. {label}: {confidence*100:.2f}%")

# -----------------------------
# 5. Apply confidence threshold
# -----------------------------
threshold = 0.70

print("\nPredictions above 70% confidence:")
found = False
for _, label, confidence in decoded:
    if confidence >= threshold:
        print(f"{label}: {confidence*100:.2f}%")
        found = True

if not found:
    print("No predictions met the confidence threshold.")

# -----------------------------
# 6. Reflection (printed output)
# -----------------------------
print("\n--- Reflection ---")
print("Confidence scores represent how certain the model is about its predictions,")
print("similar to probability in scientific measurements. However, they are not guarantees.")
print("Using a 70% threshold is like setting a detection limit in chemistry, where")
print("only signals above a certain level are considered reliable.")
print("Lower thresholds increase sensitivity (more detections but more false positives),")
print("while higher thresholds increase specificity (fewer false positives but risk missing true results).")