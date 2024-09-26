import os
import math
import numpy as np
import cv2
import pickle

def create_image_grid(image_list, num_rows=None, num_cols=None, image_size=(100, 100), space_color=(255, 255, 255), space_width=10):
    """
    Arranges a list of images into a grid format. The function resizes the images, adds optional spacing between them, 
    and calculates the appropriate number of rows and columns if not provided. It returns a single image representing 
    the grid.

    Parameters:
    - image_list (list): A list of images (as numpy arrays or PIL Image objects) to arrange in the grid.
    - num_rows (int, optional): The number of rows in the grid. If not provided, the number of rows is calculated based 
      on the total number of images and the number of columns.
    - num_cols (int, optional): The number of columns in the grid. If not provided, the number of columns is calculated 
      based on the total number of images and the number of rows.
    - image_size (tuple, optional): The size to which each image should be resized (width, height). Default is (100, 100).
    - space_color (tuple, optional): The color of the space between images, represented as an RGB tuple. Default is white (255, 255, 255).
    - space_width (int, optional): The width of the space (in pixels) between images. Default is 10.

    Returns:
    - grid_image: A single image combining all the images from the input list, arranged in the specified grid format.

    Notes:
    - If neither `num_rows` nor `num_cols` are provided, the function automatically calculates a square grid 
      based on the number of images in `image_list`.
    - The spacing between images can be customized by adjusting the `space_width` parameter.
    - The images will be resized to `image_size` before being arranged in the grid.
    """

    if num_rows is None and num_cols is None:
        num_images = len(image_list)
        num_cols = int(math.sqrt(num_images))
        num_rows = math.ceil(num_images / num_cols)
    elif num_rows is None:
        num_rows = math.ceil(len(image_list) / num_cols)
    elif num_cols is None:
        num_cols = math.ceil(len(image_list) / num_rows)

    grid_width = num_cols * (image_size[0] + space_width) - space_width
    grid_height = num_rows * (image_size[1] + space_width) - space_width
    grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    grid_image[:, :] = space_color

    for i, image in enumerate(image_list):
        image = cv2.resize(image, image_size)
        row = i // num_cols
        col = i % num_cols
        x = col * (image_size[0] + space_width)
        y = row * (image_size[1] + space_width)
        grid_image[y: y + image_size[1], x: x + image_size[0]] = image
    
    for row in range(num_rows):
        y = row * (image_size[1] + space_width) + 5
        cv2.putText(grid_image, str(row + 1), (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    for col in range(num_cols):
        x = col * (image_size[0] + space_width) + 5
        cv2.putText(grid_image, str(col + 1), (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return grid_image


def hconcat_resize(img_list, interpolation=cv2.INTER_CUBIC):
    """
    Horizontally concatenates a list of images after resizing each image to match the height of the smallest image in the list. 

    Parameters:
    - img_list (list): A list of images (as numpy arrays) to be concatenated horizontally. 
    - interpolation (int, optional): The interpolation method to be used when resizing the images. Default is 
      `cv2.INTER_CUBIC`.

    Returns:
    - concatenated_image: A single image resulting from the horizontal concatenation of the resized images.

    Notes:
    - The height of the smallest image in the list is used as the target height for resizing all other images.
    - Make sure all images in `img_list` have valid data and are loaded correctly before passing them to the function.
    
    Reference:
    - This function is taken from an example found at: https://www.geeksforgeeks.org/concatenate-images-using-opencv-in-python/
    """
    h_min = min(img.shape[0] for img in img_list)
    im_list_resize = [cv2.resize(img, (int(img.shape[1] * h_min / img.shape[0]), h_min), interpolation=interpolation) for img in img_list]

    return cv2.hconcat(im_list_resize)


def read_dict(path):
    """
    Loads a dictionary of image features from a specified file using the `pickle` module. The file should contain 
    a serialized Python dictionary, where each key-value pair represents image features or related data.

    Parameters:
    - path (str): The file path to the pickle file containing the serialized dictionary.

    Returns:
    - data_dict (dict): The dictionary of image features loaded from the file.

    Notes:
    - The function assumes the file at `path` is a valid pickle file and contains a Python dictionary.
    - If the file does not exist or is not a valid pickle file, an error will be raised during the loading process.
    """

    input_path = os.path.join(path, "feature_dictionary.pkl")
    with open(input_path, "rb") as file:
        feature_dict = pickle.load(file)

    return feature_dict

def save_dict(feature_dict, path):
    """
    Saves a dictionary of image features to a specified file using the `pickle` module. 

    Parameters:
    - feature_dict (dict): The dictionary containing image features or related data to be saved.
    - path (str): The file path where the dictionary should be saved. The file will be created or overwritten if it already exists.

    Returns:
    - None

    Notes:
    -
    """

    output_path = os.path.join(path, "feature_dictionary.pkl")
    
    # Load existing data if the file exists
    if os.path.exists(output_path):
        with open(output_path, "rb") as file:
            existing_data = pickle.load(file)
        
        existing_data.update(feature_dict)  # Update the dictionary with new features
        feature_dict = existing_data
    
    # Save the updated dictionary back to the pickle file
    with open(output_path, "wb") as file:
        pickle.dump(feature_dict, file)
    
    # print(f"[yellow]Feature dictionary saved/updated at: [bold red]{output_path}")