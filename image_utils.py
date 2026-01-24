import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.filters import roberts, prewitt, sobel

def load_and_convert_to_grayscale(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def roberts_edge_detection(gray_image):
    edges = roberts(gray_image)
    return (edges * 255).astype(np.uint8)

def prewitt_edge_detection(gray_image):
    edges = prewitt(gray_image)
    return (edges * 255).astype(np.uint8)

def sobel_edge_detection(gray_image):
    edges = sobel(gray_image)
    return (edges * 255).astype(np.uint8)

def canny_edge_detection(gray_image):
    return cv2.Canny(gray_image, threshold1=100, threshold2=200)

def process_image(image_path, output_dir="output"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    gray_image = load_and_convert_to_grayscale(image_path)
    if gray_image is None: return
        
    # Process various Edge Detection algorithms
    results = {
        "Original": cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB),
        "Grayscale": gray_image,
        "Roberts": roberts_edge_detection(gray_image),
        "Prewitt": prewitt_edge_detection(gray_image),
        "Sobel": sobel_edge_detection(gray_image),
        "Canny": canny_edge_detection(gray_image)
    }
    
    # Visualization using Matplotlib
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Edge Detection Comparison: {os.path.basename(image_path)}', fontsize=16)
    
    for ax, (title, img) in zip(axes.ravel(), results.items()):
        cmap = 'gray' if title != "Original" else None
        ax.imshow(img, cmap=cmap)
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    
    # Save the result to the output folder
    save_path = os.path.join(output_dir, f"result_{os.path.basename(image_path)}")
    plt.savefig(save_path)
    print(f"Successfully processed and saved to: {save_path}")