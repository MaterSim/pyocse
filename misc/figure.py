from PIL import Image, ImageDraw, ImageFont
import numpy as np

def combine_images_smart(image_filenames, output_filename):
    """
    Combine a list of images into a single figure, arranging them in a grid based on the number of images.

    Args:
        image_filenames (list of str): List of image filenames to combine.
        output_filename (str): Filename for the combined output image.
    """
    if not image_filenames:
        print("No input images provided.")
        return

    num_images = len(image_filenames)

    # Determine the grid dimensions for the best fit
    best_fit = None
    min_empty_cells = float('inf')
    min_ratio = float('inf')

    if num_images == 1:
        best_fit = (1, 1)
    elif num_images == 2:
        best_fit = (1, 2)
    else:
        best_fit = (int(np.ceil(num_images/3)), 3)

    print(num_images, best_fit)
    # Open all the images and get their sizes
    images = [Image.open(filename) for filename in image_filenames]
    image_width, image_height = images[0].size

    # Calculate the size of the combined image
    combined_width = best_fit[1] * image_width
    combined_height = best_fit[0] * image_height

    # Create a new image with the combined size
    combined_image = Image.new('RGB', (combined_width, combined_height))

    # Paste the images into the new image in the grid pattern
    for i, img in enumerate(images):
        row = i // best_fit[1]
        col = i % best_fit[1]
        x_offset = col * image_width
        y_offset = row * image_height
        combined_image.paste(img, (x_offset, y_offset))

        # Add file name as text
        file_name = image_filenames[i]
        draw = ImageDraw.Draw(combined_image)
        text_x = x_offset + 5 
        text_y = y_offset + img.height - 20
        draw.text((text_x, text_y), file_name, (255, 255, 255))
        print(text_x, text_y, file_name)

    # Save the combined image
    combined_image.save(output_filename)

# Example usage:
from glob import glob
image_filenames = glob('MT-*-openff/*/*.png')
image_filenames.sort()
print(image_filenames)
combine_images_smart(image_filenames, 'total.png')
