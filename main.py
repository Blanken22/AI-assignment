import numpy as np
import cv2
import matplotlib.pyplot as plt

# Define the path to the dataset image acquired from Roboflow or Google Dataset Search
image_path = 'your_salad_image.jpg' 
img = cv2.imread(image_path)

if img is not None:
    # Convert the loaded image from OpenCV's default BGR color space to standard RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Reshape the three-dimensional image array into a two-dimensional list of pixels
    pixels = img_rgb.reshape((-1, 3))
    
    # Convert the pixel values to 32-bit floating point numbers for distance calculations
    pixels = np.float32(pixels)

    # Define the termination criteria for the K-means iterative algorithm
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    
    # Define the number of clusters to segment the image into
    k = 4
    flags = cv2.KMEANS_RANDOM_CENTERS

    # Execute the K-means clustering algorithm on the pixel data
    ret, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, flags)

    # Convert the resulting mathematically derived centroids back into 8-bit color values
    centers = np.uint8(centers)
    
    # Prepare a list of empty black images to hold the reconstructed clusters
    segmented_imgs = [np.zeros_like(img_rgb) for _ in range(k)]

    # Initialize variables to track the autonomous self-recognition heuristic
    greenest_cluster_idx = -1
    max_green_ratio = -1

    # Iterate through the clusters to apply masks and execute the self-recognition logic
    for i in range(k):
        # Create a spatial mask mapping the flat labels back to the original image dimensions
        cluster_mask = (labels == i).reshape(img_rgb.shape[:2])
        
        # Apply the mask to isolate the pixels belonging to the current cluster
        segmented_imgs[i][cluster_mask] = img_rgb[cluster_mask]

        # Extract the Red, Green, and Blue values from the current centroid
        r, g, b = centers[i]
        
        # Calculate the total color intensity to prevent division by zero errors
        total_color = int(r) + int(g) + int(b)
        
        if total_color > 0:
            # Calculate the proportion of green light relative to the total color
            green_ratio = g / total_color
            
            # Determine if this cluster holds the highest concentration of green thus far
            if green_ratio > max_green_ratio:
                max_green_ratio = green_ratio
                greenest_cluster_idx = i

    # Initialize the Matplotlib visualization environment with a large figure size
    plt.figure(figsize=(15, 10))
    
    # Configure the first subplot to display the original, unaltered image
    plt.subplot(2, 3, 1)
    plt.imshow(img_rgb)
    plt.title('Original Salad Image')
    plt.axis('off')

    # Iterate through the segmented images to populate the remaining subplots
    for i in range(k):
        plt.subplot(2, 3, i + 2)
        plt.imshow(segmented_imgs[i])
        
        # Apply conditional labeling based on the autonomous recognition heuristic
        if i == greenest_cluster_idx:
            plt.title(f'Cluster {i + 1} (Autonomous Recognition: Salad)')
        else:
            plt.title(f'Cluster {i + 1}')
            
        plt.axis('off')

    # Adjust the graphical layout to ensure titles and images do not overlap
    plt.tight_layout()
    plt.show()
else:
    print("System Error: The specified dataset image could not be located or loaded. Please verify the file path.")