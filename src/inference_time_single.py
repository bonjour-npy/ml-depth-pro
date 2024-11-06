# type: ignore
from PIL import Image
import depth_pro
from matplotlib import pyplot as plt
import os


image_path = "./data/example.jpg"
# Load model and preprocessing transform
model, transform = depth_pro.create_model_and_transforms()
model.eval()

# Load and preprocess an image.
image, _, f_px = depth_pro.load_rgb(image_path)
image = transform(image)

# Run inference.
prediction = model.infer(image, f_px=f_px)
depth = prediction["depth"]  # Depth in [m].
focallength_px = prediction["focallength_px"]  # Focal length in pixels.

# Save the depth map
output_path = os.path.join("./output", os.path.basename(image_path))
plt.imsave(output_path, depth, cmap="plasma")
print(f"Depth map saved to {output_path}")
