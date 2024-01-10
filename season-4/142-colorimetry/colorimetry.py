"""
Colorimetry Applied to Cannabis
Copyright (c) 2023 Cannlytics

Authors: Keegan Skeate <https://github.com/keeganskeate>
Created: 1/6/2024
Updated: 1/7/2024
License: MIT License <https://github.com/cannlytics/cannabis-data-science/blob/main/LICENSE>
"""
# Standard imports:
import ast
import os
from time import sleep
import matplotlib.pyplot as plt
import numpy as np
import requests

# External imports:
import cv2
from PIL import Image
from cannlytics.utils import snake_case
import pandas as pd
from rembg import remove
import numpy as np
from skimage import color
from sklearn.cluster import KMeans


# === Setup ===

# Setup plotting style.
plt.style.use('fivethirtyeight')
plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 24,
})


def remove_bg(input_path: str, output_path: str) -> None:
    """Convert a video file to another video with a transparent background.
    
    Args:
        input_path (str): The path to the input image file.
        output_path (str): The path to save the output image with a transparent background.
    """
    input = Image.open(input_path)
    output = remove(input)
    output.save(output_path)



# === Read the data ===

# Read SC lab's flower results.
results = pd.read_excel('data/augmented-ca-lab-results-sclabs-2024-01-03-15-26-35.xlsx')
# emerald_cup = results.loc[results['producer'].str.contains('Emerald Cup')]
flower_types = [
    'Flower, Inhalable',
    'Flower, Inhaled Product',
    'Flower, Product Inhalable',
]
flower = results.loc[results['product_type'].isin(flower_types)]
emerald_cup = flower.loc[flower['producer'].astype(str).str.contains('Emerald Cup')]

# Read Glass House flower results.


# Read Flower Company flower results.
# results = pd.read_excel(r"D:\data\california\lab_results\ca-lab-results-2023-12-30.xlsx")


# === Image management ===

# Download the images to an image dir.
image_files = []
image_dir = 'D://data/california/lab_results/images/sclabs'
for index, row in flower.iterrows():
    images = ast.literal_eval(row['images'])
    if images:
        coa_id = row['coa_id'].split('-')[0].strip()
        product_name = row['product_name']
        slug = snake_case(product_name)
        filename = f'{image_dir}/{coa_id}-{slug}.jpg'
        if os.path.exists(filename):
            image_files.append(filename)
            continue
        image_url = images[0]['url']
        response = requests.get(image_url)
        if response.status_code == 200:
            print(f"Downloaded: {image_url}")
            with open(filename, 'wb') as f:
                f.write(response.content)
        else:
            print(f"Failed to download: {image_url}")
        image_files.append(filename)
        sleep(1)

# Crop the images.
cropped_images = []
for image_file in image_files:
    cropped_file = image_file.replace('.jpg', '-cropped.png')
    if os.path.exists(cropped_file):
        cropped_images.append(cropped_file)
        continue
    remove_bg(image_file, cropped_file)
    cropped_images.append(cropped_file)
    print(f'Cropped: {cropped_file}')


# DEV:
# cropped_images = [os.path.join(image_dir, x) for x in os.listdir(image_dir) if x.endswith('-cropped.png')]


# === Color Analysis ===

def calculate_purpleness(rgb, how='scale'):
    """Purple is dominant in red and blue channels, and low in green.
    Args:
        rgb (list): A list of RGB values.
        how (str): How to calculate purpleness. Options are 'scale' and 'normalized'.
        Scale will return a value between 0 and 1. Normalized will return a value between -1 and 1.
    
    Returns:
        float: The purpleness score.
    
    Note: Adjust the formula for other shades of purple.
    """
    purpleness = (rgb[0] + rgb[2]) - 2*rgb[1]
    if how == 'scale':
        purpleness = purpleness + 510
        purpleness = purpleness / 1020
    elif how == 'normalized':
        purpleness = purpleness / 510
    return purpleness


def calculate_colorfulness(image):
    """Calculate the colorfulness metric described in:
    Hasler and Süsstrunk, "Measuring colorfulness in natural images."""
    # Split the image into its respective RGB components.
    (R, G, B) = cv2.split(image.astype("float"))

    # Compute rg = R - G
    rg = np.absolute(R - G)

    # Compute yb = 0.5 * (R + G) - B
    yb = np.absolute(0.5 * (R + G) - B)

    # Compute the mean and standard deviation of both `rg` and `yb`.
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))

    # Combine the mean and standard deviation.
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))

    # Derive the "colorfulness" metric and return it.
    return stdRoot + (0.3 * meanRoot)


def calculate_colourfulness(image, metric='M3'):
    # Convert the image from RGB to CIELab color space
    lab_image = color.rgb2lab(image)
    l, a, b = lab_image[:, :, 0], lab_image[:, :, 1], lab_image[:, :, 2]

    # Compute standard deviations and means in the CIELab space
    sigma_a, sigma_b = np.std(a), np.std(b)
    mu_a, mu_b = np.mean(a), np.mean(b)
    sigma_ab = np.sqrt(sigma_a**2 + sigma_b**2)
    mu_ab = np.sqrt(mu_a**2 + mu_b**2)
    # Aab = sigma_a * sigma_b

    # Compute Chroma and Saturation
    Chroma = np.sqrt(a**2 + b**2)
    _, mu_C = np.std(Chroma), np.mean(Chroma)
    # Saturation = Chroma / l
    # sigma_S, mu_S = np.std(Saturation), np.mean(Saturation)

    # Convert the image to a simplified opponent color space
    R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    rg = R - G
    yb = 0.5 * (R + G) - B
    sigma_rg, sigma_yb = np.std(rg), np.std(yb)
    mu_rg, mu_yb = np.mean(rg), np.mean(yb)
    sigma_rg_yb = np.sqrt(sigma_rg**2 + sigma_yb**2)
    mu_rg_yb = np.sqrt(mu_rg**2 + mu_yb**2)

    # Return the desired metric.
    if metric == 'M1':
        return sigma_ab + 0.37 * mu_ab
    elif metric == 'M2':
        return sigma_ab + 0.94 * mu_C
    elif metric == 'M3':
        return sigma_rg_yb + 0.3 * mu_rg_yb
    else:
        raise ValueError('Unknown metric: %s' % metric)


def rgb_to_xyz(rgb):
    # Convert the RGB values to the range [0, 1]
    rgb = rgb / 255.0
    # Apply the RGB to XYZ conversion
    xyz = np.dot(rgb, [[0.412453, 0.357580, 0.180423],
                       [0.212671, 0.715160, 0.072169],
                       [0.019334, 0.119193, 0.950227]])
    return xyz

def xyz_to_chromaticity(xyz):
    # Avoid division by zero
    total = np.maximum(xyz[:, :, 0] + xyz[:, :, 1] + xyz[:, :, 2], 1e-12)
    x = xyz[:, :, 0] / total
    y = xyz[:, :, 1] / total
    return x, y


def plot_chromaticity_diagram(x, y):
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Generate a mesh to fill the color space
    xx, yy = np.meshgrid(np.linspace(0, 0.8, 400), np.linspace(0, 0.9, 400))
    zz = 1 - xx - yy

    # Keep only points inside the color gamut
    inside = (xx >= 0) & (yy >= 0) & (zz >= 0)

    # Calculate the RGB values
    xyY = np.stack([xx, yy, np.ones_like(xx)], axis=-1)
    XYZ = xyY / xyY[:, :, 1:2]
    RGB = np.matmul(XYZ, np.linalg.inv([[0.412453, 0.357580, 0.180423],
                                        [0.212671, 0.715160, 0.072169],
                                        [0.019334, 0.119193, 0.950227]]).T)
    # Clip values to valid range
    RGB = np.clip(RGB, 0, 1)

    # Plot the colors
    ax.imshow(RGB, extent=(0, 0.8, 0, 0.9), origin='lower', aspect='auto')

    # Plot the spectral locus (approximate)
    wavelengths = np.linspace(380, 780, 400)
    spectral_x = 0.01 * wavelengths
    spectral_y = 0.01 * (780 - wavelengths)
    spectral_y /= (spectral_x + spectral_y + 0.01 * (780 - 2 * wavelengths))
    spectral_x /= (spectral_x + spectral_y + 0.01 * (780 - 2 * wavelengths))
    ax.plot(spectral_x, spectral_y, color='white')

    # Plot the image points
    ax.scatter(x, y, c='blue', alpha=0.1)

    # Labeling the plot
    ax.set_xlim([0, 0.8])
    ax.set_ylim([0, 0.9])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Chromaticity Diagram')
    plt.grid(True)
    plt.show()


def find_dominant_color(image, k=1, image_processing_size=None):
    """
    Find the dominant color in an image.
    
    :param image: Image file path
    :param k: Number of clusters for k-means
    :param image_processing_size: Tuple (width, height), size to which the image is resized for processing
    :return: Dominant color in RGB
    """
    # Read the image
    image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
    
    # Resize image if new dimensions are provided for faster processing
    if image_processing_size is not None:
        image = cv2.resize(image, image_processing_size, interpolation=cv2.INTER_AREA)
    
    # Check if the image has an alpha channel
    if image.shape[-1] == 4:
        # Create a mask and apply to only consider non-transparent pixels
        alpha_channel = image[:, :, 3]
        mask = alpha_channel > 0
        image = image[:, :, :3][mask].reshape(-1, 3)
    else:
        # Flatten the image array and convert to float32 for k-means
        image = image.reshape((-1, 3)).astype(np.float32)
    
    # Apply k-means clustering to find k colors
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(image)
    
    # Convert the dominant color to integer values
    dominant_color = kmeans.cluster_centers_[0].astype(int)
    
    # Ensure the color is within valid range
    dominant_color = np.clip(dominant_color, 0, 255)
    
    return tuple(dominant_color)


def get_dominant_color(image, k=1, mask=None):
    """
    Find the dominant color(s) in an image.
    :param image: Image to analyze
    :param k: Number of dominant colors to find
    :param mask: Optional mask to apply on the image
    :return: List of dominant colors in BGR format
    """
    # Check if mask is given
    if mask is not None:
        image = cv2.bitwise_and(image, image, mask=mask)
        # Convert the masked image to a list of pixels
        pixels = image[mask > 0].reshape(-1, 3)
    else:
        pixels = image.reshape(-1, 3)
        
    pixels = np.float32(pixels)
    
    # Define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, palette = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Find the most frequent label
    _, counts = np.unique(labels, return_counts=True)
    dominant = palette[np.argmax(counts)]
    
    return dominant.astype(int)

def plot_histogram(image, mask=None):
    """
    Plot the color histogram of an image.
    :param image: Image to analyze
    :param mask: Optional mask to apply on the image
    """
    # Convert to RGB for matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply mask if provided
    if mask is not None:
        image_rgb = image_rgb[mask > 0]
    
    # Flatten the image array and get the channels
    r, g, b = cv2.split(image_rgb)
    
    fig, (ax_r, ax_g, ax_b) = plt.subplots(1, 3, figsize=(15, 5))
    for ax, channel, color in zip((ax_r, ax_g, ax_b), (r, g, b), ('Red', 'Green', 'Blue')):
        ax.hist(channel.ravel(), bins=256, color=color.lower(), alpha=0.5)
        ax.set_xlim([0, 256])
        ax.set_title(f'{color} Histogram')
    plt.show()


# === DEV ===

# # Load the image
# image = cv2.imread(image_file)

# # Check if there's an alpha channel
# if image.shape[-1] == 4:
#     # Create a mask for pixels where the alpha channel is not transparent
#     mask = image[:, :, 3] > 0
#     image = image[:, :, :3]  # Drop the alpha channel for color analysis
# else:
#     mask = None

# # Plot the histogram
# plot_histogram(image, mask)

# # Calculate the dominant color
# dominant_color = get_dominant_color(image, mask=mask)
# print("Dominant color (BGR):", dominant_color)


# Replace 'image_path' with the path of your image
# dominant_color = find_dominant_color(image_file)

# DEV:
# # Assuming cropped_images is a list of image file paths
# for image_file in cropped_images[30:100]:  # Just taking first 3 for demonstration

#     # image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)  # Read the image with alpha channel
#     # # Check if there is an alpha channel
#     # if image.shape[-1] == 4:
#     #     # Split the image into RGBA channels
#     #     b_channel, g_channel, r_channel, alpha_channel = cv2.split(image)
        
#     #     # Create a mask where the alpha channel is not transparent
#     #     mask = alpha_channel > 0
        
#     #     # Use the mask to select only the non-transparent pixels
#     #     non_transparent_pixels = image[mask]

#     #     # Compute the mean color of the non-transparent pixels
#     #     mean_color = cv2.mean(image[:, :, :3], mask=mask.astype(np.uint8))[:3]
#     # else:
#     #     # No alpha channel, so just compute the mean color directly
#     #     cropped_img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     #     mean_color = cropped_img_rgb.mean(axis=(0, 1))

#     # print(f"Mean color (ignoring transparency): {mean_color}")
#     color = find_dominant_color(image_file)

#     # Normalize the RGB values to the range [0, 1] for Matplotlib
#     mean_color_normalized = np.array(color) / 255

#     # Create a figure and a subplot with a title
#     fig, ax = plt.subplots()
#     ax.set_title('Mean Color of the Image')
#     ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=mean_color_normalized))
#     ax.axis('off')  # Hide the axes
#     plt.show()

#     # FIXME:
#     # image = cv2.imread(image_path)
#     # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
#     # xyz = rgb_to_xyz(image)
#     # x, y = xyz_to_chromaticity(xyz)
#     # plot_chromaticity_diagram(x.flatten(), y.flatten())




# Assign purpleness.
purple_scores = {}
for image_file in cropped_images:
    coa_id = os.path.split(image_file)[-1].split('-')[0]
    if coa_id in purple_scores:
        continue
    image = cv2.imread(image_file)
    try:
        cropped_img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except:
        continue
    mean_color = cropped_img_rgb.mean(axis=0).mean(axis=0)
    # color_scores[coa_id] = mean_color
    purple_scores[coa_id] = calculate_purpleness(mean_color)
    print(f'Purpleness for {coa_id}: {purple_scores[coa_id]}')

# Assign colourfulness.
colorfulness_scores = {}
for image_file in cropped_images:
    coa_id = os.path.split(image_file)[-1].split('-')[0]
    if coa_id in colorfulness_scores:
        continue
    image = cv2.imread(image_file)
    try:
        cropped_img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except:
        continue
    colorfulness = calculate_colourfulness(cropped_img_rgb)
    # colorfulness = calculate_colorfulness(cropped_img_rgb)
    colorfulness_scores[coa_id] = colorfulness
    print(f'Color score for {coa_id}: {colorfulness_scores[coa_id]}')

# Merge color scores with the data.
flower['id'] = flower['coa_id'].apply(lambda x: x.split('-')[0].strip())
flower['purpleness'] = flower['id'].map(purple_scores)
flower['colorfulness_score'] = flower['id'].map(colorfulness_scores)


# === Color Analysis Visualizations ===

# === The Cannlytics Most Purple Flower Award ===

# Top 10 most purple products in 2022.
emerald_cup_2022 = flower.loc[flower['producer'].astype(str).str.contains('Emerald Cup 2022')]
purple = emerald_cup_2022.sort_values(by='purpleness', ascending=False)
purple = purple.loc[purple['purpleness'] < 5]
print('Top 10 Most Purple Flowers 2022')
print(purple.head(10)[['coa_id', 'product_name', 'purpleness']])
most_purple = purple.loc[purple['purpleness'] == purple['purpleness'].max()]
print(f'Most Purple Flower 2022: {most_purple.iloc[0]["product_name"]}, {most_purple.iloc[0]["purpleness"]}')

# Top 10 most purple products in 2023.
emerald_cup_2023 = flower.loc[flower['producer'].astype(str).str.contains('Emerald Cup 2023')]
purple = emerald_cup_2023.sort_values(by='purpleness', ascending=False)
purple = purple.loc[purple['purpleness'] < 5]
print('Top 10 Most Purple Flowers 2023')
print(purple.head(10)[['coa_id', 'product_name', 'purpleness']])
most_purple = purple.loc[purple['purpleness'] == purple['purpleness'].max()]
print(f'Most Purple Flower 2023: {most_purple.iloc[0]["product_name"]}, {most_purple.iloc[0]["purpleness"]}')

# Visualize the purpleness scores.
plt.figure(figsize=(15, 8))
plt.hist(
    emerald_cup_2022['purpleness'].loc[emerald_cup_2022['purpleness'] < 5],
    bins=30,
    alpha=0.7,
    label='2022',
    color='violet',
)
plt.hist(
    emerald_cup_2023['purpleness'].loc[emerald_cup_2023['purpleness'] < 5],
    bins=30,
    alpha=0.55,
    label='2023',
    color='darkviolet',
)
plt.xlabel('Purpleness Score')
plt.ylabel('Count')
plt.title('Flower Purpleness (Emerald Cup 2022 vs 2023)')
plt.legend()
# plt.savefig('./presentation/images/emerald-cup-purple-scores.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

# Visualize the colorfulness scores.
plt.figure(figsize=(15, 8))
plt.hist(
    emerald_cup_2022['colorfulness_score'],
    bins=30,
    alpha=0.7,
    label='2022',
    color='green',
)
plt.hist(
    emerald_cup_2023['colorfulness_score'],
    bins=30,
    alpha=0.7,
    label='2023',
    color='orangered',
)
plt.xlabel('Colorfulness Score')
plt.ylabel('Count')
plt.title('Flower Colorfulness (Emerald Cup 2022 vs 2023)')
plt.legend()
# plt.savefig('./presentation/images/emerald-cup-colorfulness-scores.pdf', dpi=300, bbox_inches='tight', transparent=True)
plt.show()

# Top 10 most colorful products in 2022.
emerald_cup_2022 = flower.loc[flower['producer'].astype(str).str.contains('Emerald Cup 2022')]
colorful_2022 = emerald_cup_2022.sort_values(by='colorfulness_score', ascending=False)
print('Top 10 Most Colorful Flowers 2022')
print(colorful_2022.head(10)[['coa_id', 'product_name', 'colorfulness_score']])
most_colorful_2022 = colorful_2022.iloc[0]
print(f'Most Colorful Flower 2022: {most_colorful_2022["product_name"]}, Colorfulness Score: {most_colorful_2022["colorfulness_score"]}')

# Top 10 most colorful products in 2023.
emerald_cup_2023 = flower.loc[flower['producer'].astype(str).str.contains('Emerald Cup 2023')]
colorful_2023 = emerald_cup_2023.sort_values(by='colorfulness_score', ascending=False)
print('Top 10 Most Colorful Flowers 2023')
print(colorful_2023.head(10)[['coa_id', 'product_name', 'colorfulness_score']])
most_colorful_2023 = colorful_2023.iloc[0]
print(f'Most Colorful Flower 2023: {most_colorful_2023["product_name"]}, Colorfulness Score: {most_colorful_2023["colorfulness_score"]}')

# # The Cannlytics Most Colorful Flower Award (Assuming flower includes data from both 2022 and 2023)
# most_colorful_overall = flower.loc[flower['colorfulness_score'] == flower['colorfulness_score'].max()]
# print(f'Most Colorful Flower Overall: {most_colorful_overall.iloc[0]["product_name"]}, Colorfulness Score: {most_colorful_overall.iloc[0]["colorfulness_score"]}')

# # Top 10 most colorful products overall.
# colorful_overall = flower.sort_values(by='colorfulness_score', ascending=False).head(10)
# print('Top 10 Most Colorful Flowers Overall')
# print(colorful_overall[['coa_id', 'product_name', 'colorfulness_score']])
