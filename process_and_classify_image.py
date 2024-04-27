import cv2
import numpy as np
import os
from contour_utils import find_colors, clean_and_combine_masks, classify_long_short_or_unknown, classify_long_short_or_unknown_black_background
from display import display_image, display_mask
from file_utils import sanitize_filename, create_save_path
from image_utils import preprocess_image
from checkImageBackground import check_background_color
import argparse


def check_vertical_alignment(index_above, index_below, contours, max_vertical_distance=200):
    xa, ya, wa, ha = cv2.boundingRect(contours[index_above])
    xb, yb, wb, hb = cv2.boundingRect(contours[index_below])

    # Bottom of the contour above
    bottom_above = ya + ha
    # Top of the contour below
    top_below = yb

    # Check vertical proximity
    if not (top_below > bottom_above and (top_below - bottom_above) < max_vertical_distance):
        return False

    # Check horizontal overlap by finding the horizontal range intersection
    # The horizontal range of the contour above
    horizontal_range_above = set(range(xa, xa + wa))
    # The horizontal range of the contour below
    horizontal_range_below = set(range(xb, xb + wb))

    # If there's no intersection in the horizontal ranges, they are not aligned
    if not horizontal_range_above.intersection(horizontal_range_below):
        return False

    return True


def auto_check_vertical_contours(contours, max_vertical_distance=200):
    contour_relationships = {index: False for index in range(len(contours))}
    # print(len(contours))
    for i in range(len(contours)):
        for j in range(len(contours)):
            if i != j:  # Ensure we're not comparing the same contour
                if check_vertical_alignment(i, j, contours, max_vertical_distance):
                    # Mark as having a valid contour beneath
                    contour_relationships[i] = True
                    print(f"Contour {i} is above Contour {j}")

    return [index for index, has_beneath in contour_relationships.items() if has_beneath]


def find_vertically_aligned_contours(all_contours, max_vertical_distance=200):
    vertically_aligned_pairs = []
    # Exclude the last contour to prevent index out of range
    for i, contour_a in enumerate(all_contours[:-1]):
        # Only check contours below contour_a
        for j, contour_b in enumerate(all_contours[i+1:]):
            if check_vertical_alignment(i, j, all_contours, max_vertical_distance):
                vertically_aligned_pairs.append((i, j))

    return vertically_aligned_pairs


def include_additional_contours(vertically_aligned_pairs, all_contours):
    # Initialize a set to keep track of individual indices
    horizontally_included_contours = set()
    for index_above, index_below in vertically_aligned_pairs:
        xa, ya, wa, ha = cv2.boundingRect(all_contours[index_above])
        xb, yb, wb, hb = cv2.boundingRect(all_contours[index_below])

        # Calculate horizontal range from the leftmost to the rightmost edges
        left_bound = min(xa, xb)
        right_bound = max(xa + wa, xb + wb)

        # Check for contours below or above the index_below contour that are within the horizontal bounds
        for k, contour_c in enumerate(all_contours):
            if k != index_above and k != index_below:
                xc, yc, wc, hc = cv2.boundingRect(contour_c)
                # Check if contour_c is horizontally within the bounds of the aligned pair
                if xc < right_bound and (xc + wc) > left_bound:
                    # If contour_c is below index_below and within horizontal bounds
                    if yc > yb + hb:
                        horizontally_included_contours.add(k)
                    # If contour_c is above index_above and within horizontal bounds
                    if yc + hc < ya:
                        horizontally_included_contours.add(k)

    return list(horizontally_included_contours)


def filter_by_edge_count(contours, max_edges=20):
    filtered_contours = []
    for contour in contours:
        # Approximate the contour to a polygon
        # Adjust epsilon as needed
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        # print("Number of edges:", len(approx))

        # If the number of edges is less than or equal to the threshold, keep it
        if len(approx) <= max_edges:
            filtered_contours.append(contour)
    return filtered_contours


def clip_contours_to_right(contours, image_width, right_ratio=0.6):
    clipped_contours = []

    # Define the x-coordinate threshold for the right 60%
    right_threshold = image_width * (1 - right_ratio)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Check if any part of the contour is within the right 60%
        if (x + w) > right_threshold:
            # Clip the contour to the right 60% region
            new_x = max(x, right_threshold)
            new_w = x + w - new_x

            if new_w > 0:
                # Adjust contour points to only include those in the right 60%
                new_contour = [
                    point for point in contour
                    if point[0][0] >= right_threshold
                ]
                clipped_contours.append(np.array(new_contour, dtype=np.int32))

    return clipped_contours


def filter_by_minimum_area(contours, min_area):
    """
    Filters contours based on the minimum area.
    Only keeps contours with an area greater than or equal to the specified minimum area.
    """
    filtered_contours = [
        contour for contour in contours if cv2.contourArea(contour) >= min_area]
    return filtered_contours

def match_contour_pairs(contours_with_centers, x_distance_threshold=10, width_tolerance=20):
    """
    Find pairs of contours with centers that are close on the X-axis
    and have similar widths.
    """
    matched_pairs = []  # To store matched pairs of contours

    # Loop through all contour combinations
    for i in range(len(contours_with_centers)):
        for j in range(i + 1, len(contours_with_centers)):
            # Get the center points and widths of the two contours
            contour1 = contours_with_centers[i]
            contour2 = contours_with_centers[j]

            x_distance = abs(contour1["leftmost_x"] - contour2["leftmost_x"])

            # Check the difference in widths
            width_difference = abs(contour1["width"] - contour2["width"])

            # print(f"Contour {i+1} & {j+1} - X Distance: {x_distance}, Width Difference: {width_difference}")

            # If the distance and width are within the specified thresholds
            if x_distance <= x_distance_threshold and width_difference <= width_tolerance:
                # Add the pair to the list of matched pairs
                matched_pairs.append((contour1, contour2))

    return matched_pairs  # Return the list of matched pairs

# Define a function to check if a contour is valid
def is_valid_contour(contour):
    # A valid contour must have at least three points
    return isinstance(contour, np.ndarray) and contour.ndim == 3 and contour.shape[1] == 1 and contour.shape[2] == 2

def find_most_rectangular(contours):
    most_rectangular_contour = None
    most_rectangular_score = float('inf')  # Initialize with a high value
    most_rectangular_contour_number = None

    # Loop through contours to find the most rectangular
    for i, contour in enumerate(contours):
        if is_valid_contour(contour):
            # Get the bounding rectangle
            x, y, width, height = cv2.boundingRect(contour)

            # Extract the top-left and bottom-left corners
            top_left = (x, y)
            bottom_left = (x, y + height)

            # Calculate the horizontal distance between these points
            horizontal_distance = abs(top_left[0] - bottom_left[0])

            # Compute a score based on the height-to-width ratio and horizontal distance
            rectangularity_score = horizontal_distance / width  # Closer to 0 is better

            # Ensure the contour has a significant width-to-height ratio
            if width >= 500 and height > 0 and width > height:
                if rectangularity_score < most_rectangular_score:
                    most_rectangular_score = rectangularity_score
                    most_rectangular_contour = contour
                    most_rectangular_contour_number = i + 1  # Keep track of contour number

    return most_rectangular_contour_number

def is_right_angle(v1, v2, tolerance=0.1):
    dot_product = np.dot(v1, v2)
    return np.isclose(dot_product, 0, atol=tolerance)

# Function to validate if a set of corners form a rectangle
def is_rectangle(top_left, top_right, bottom_left, bottom_right):
    # Calculate the vectors for each side
    vector_top = np.array(top_right) - np.array(top_left)
    vector_bottom = np.array(bottom_right) - np.array(bottom_left)
    vector_left = np.array(bottom_left) - np.array(top_left)
    vector_right = np.array(bottom_right) - np.array(top_right)
    
    # Calculate the distances (lengths of the sides)
    length_top = np.linalg.norm(vector_top)
    length_bottom = np.linalg.norm(vector_bottom)
    length_left = np.linalg.norm(vector_left)
    length_right = np.linalg.norm(vector_right)

    # Check if opposite sides are roughly equal in length
    opposite_sides_equal = (
        np.isclose(length_top, length_bottom, atol=5) and
        np.isclose(length_left, length_right, atol=5)
    )
    
    # Check if adjacent sides are perpendicular
    perpendicular_sides = (
        is_right_angle(vector_top, vector_left) and
        is_right_angle(vector_left, vector_bottom) and
        is_right_angle(vector_bottom, vector_right)
    )
    
    return opposite_sides_equal and perpendicular_sides

# def draw_rectangles(image, contours_with_corners):
#     rectangle_image = image.copy()

#     for contour_data in contours_with_corners:
#         # Get the corners
#         top_left = contour_data['top_left']
#         top_right = contour_data['top_right']
#         bottom_left = contour_data['bottom_left']
#         bottom_right = contour_data['bottom_right']

#         # Check if it's a valid rectangle
#         if is_rectangle(top_left, top_right, bottom_left, bottom_right):
#             # Draw the rectangle
#             cv2.rectangle(
#                 rectangle_image,
#                 tuple(top_left),
#                 tuple(bottom_right),
#                 (0, 255, 0),  # Green color for the rectangle
#                 2
#             )
#         else:
#             # Mark it with a different color to indicate it's not a valid rectangle
#             cv2.rectangle(
#                 rectangle_image,
#                 tuple(top_left),
#                 tuple(bottom_right),
#                 (255, 0, 0),  # Red for invalid rectangles
#                 2
#             )
    
#     return rectangle_image

def draw_rectangles(image, contours_with_corners):
    rectangle_image = image.copy()

    font_scale = 1.5  # Increase this for a larger font size
    font_thickness = 2
    contour_info = []

    # Label contours with their number and check for nested rectangles
    for contour_data in contours_with_corners:
        contour_number = contour_data["contour_number"]

        # Get the corners
        top_left = contour_data['top_left']
        top_right = contour_data['top_right']
        bottom_left = contour_data['bottom_left']
        bottom_right = contour_data['bottom_right']

        # Check if it's a valid rectangle by using all four corners
        if is_rectangle(top_left, top_right, bottom_left, bottom_right):
            # Draw the rectangle
            cv2.rectangle(
                rectangle_image,
                tuple(top_left),
                tuple(bottom_right),
                (0, 255, 0),  # Green color for valid rectangles
                2
            )

            # Add contour number as a label
            label_position = (top_left[0], top_left[1] - 10)  # Slightly above the top-left corner
            cv2.putText(
                rectangle_image,
                f"{contour_number}",
                label_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,  # Use the updated font scale
                (0, 255, 0),  # Green text
                font_thickness  # Use the updated font thickness
            )
        else:
            # Mark with a different color for invalid rectangles
            cv2.rectangle(
                rectangle_image,
                tuple(top_left),
                tuple(bottom_right),
                (255, 0, 0),  # Red color for invalid rectangles
                font_thickness
            )

        # Check for nested rectangles
        is_nested = any(
            other_contour_data['contour_number'] != contour_number and
            other_contour_data['top_left'][0] >= top_left[0] and
            other_contour_data['top_left'][1] >= top_left[1] and
            other_contour_data['bottom_right'][0] <= bottom_right[0] and
            other_contour_data['bottom_right'][1] <= bottom_right[1]
            for other_contour_data in contours_with_corners
        )

        # Label nested rectangles
        if is_nested:
            nested_label_position = (top_left[0], top_left[1] - 25)  # Higher to indicate nesting
            cv2.putText(
                rectangle_image,
                f"{contour_number} (Nested)",
                nested_label_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),  # White text for nested labels
                font_thickness
            )
        # Store contour information for debugging
        contour_info.append({
            'contour_number': contour_number,
            'is_nested': is_nested,
            'top_left': top_left,
            'bottom_right': bottom_right,
        })
    
    # Display contour information in a table format
    print("Contour | Top Left | Bottom Right | Nested")
    print("------------------------------------------")
    for info in contour_info:
        print(
            f"{info['contour_number']}     | "
            f"{info['top_left']}     | "
            f"{info['bottom_right']}     | "
            f"{'Yes' if info['is_nested'] else 'No'}"
        )
    
    return rectangle_image


# Function to calculate if the contour fills enough of the rectangle area
def is_contour_filling_rectangle(contour, rectangle_area, minimum_fill_ratio=0.7):
    contour_area = cv2.contourArea(contour)
    fill_ratio = contour_area / rectangle_area
    return fill_ratio >= minimum_fill_ratio

# Function to approximate and trim contours to remove extremities
def simplify_contour(contour, epsilon=0.05):
    """
    Approximate the contour to remove extremities and smooth it.
    """
    return cv2.approxPolyDP(contour, epsilon * cv2.arcLength(contour, True), True)

def filter_rectangular_contours(contours_with_corners, minimum_fill_ratio=0.7):
    valid_contours = []

    # Sort contours by their top-left corner's x-coordinate to detect nesting
    contours_with_corners = sorted(contours_with_corners, key=lambda c: c['top_left'][0])

    # Initialize a list to track whether each contour is nested
    nested_info = []  # Store nested status and other info for each contour

    for i, outer_contour_data in enumerate(contours_with_corners):
        is_nested = False
        top_left = outer_contour_data['top_left']
        bottom_right = outer_contour_data['bottom_right']

        # Calculate the rectangle area
        rect_width = bottom_right[0] - top_left[0]
        rect_height = bottom_right[1] - top_left[1]
        rectangle_area = rect_width * rect_height

        outer_contour = outer_contour_data['contour']
        outer_contour_area = cv2.contourArea(outer_contour)
        # outer_contour_approx = approximate_contour(outer_contour)

        # outer_contour_area = cv2.contourArea(outer_contour_approx)

        # Check if this contour encompasses another (is nested)
        for inner_contour_data in contours_with_corners:
            if inner_contour_data['contour_number'] != outer_contour_data['contour_number']:
                inner_top_left = inner_contour_data['top_left']
                inner_bottom_right = inner_contour_data['bottom_right']

                if (inner_top_left[0] >= top_left[0] and
                    inner_top_left[1] >= top_left[1] and
                    inner_bottom_right[0] <= bottom_right[0] and
                    inner_bottom_right[1] <= bottom_right[1]):
                    
                    # Add the inner contour's area to the outer contour's area
                    inner_area = cv2.contourArea(inner_contour_data['contour'])
                    outer_contour_area += inner_area
                    is_nested = True
                    break  # No need to check further if nesting is detected

        # Calculate the fill ratio with nested areas
        fill_ratio = outer_contour_area / rectangle_area

        # Check the fill ratio and add to valid contours if it meets the threshold
        if fill_ratio >= minimum_fill_ratio:
            valid_contours.append(outer_contour_data)

        # Store contour information with the calculated fill ratio and nested status
        nested_info.append({
            'contour_number': outer_contour_data['contour_number'],
            'rectangle_area': rectangle_area,
            'outer_contour_area': outer_contour_area,
            'fill_ratio': fill_ratio,
            'is_nested': is_nested
        })

    # Display the nested information in a table format
    print("Contour | Rectangle Area | Outer Contour Area | Fill Ratio | Nested")
    print("----------------------------------------------------------")
    for info in nested_info:
        print(
            f"{info['contour_number']}     | "
            f"{info['rectangle_area']}     | "
            f"{info['outer_contour_area']}     | "
            f"{info['fill_ratio']:.2f}     | "
            f"{'Yes' if info['is_nested'] else 'No'}"
        )

    return valid_contours

# Example usage to filter valid rectangular contours
# valid_contours_with_corners = filter_rectangular_contours(contours_with_corners, minimum_fill_ratio=0.7)

# Function to draw rectangles based on the filtered contours
def draw_filtered_rectangles(image, contours_with_corners):
    filtered_image = image.copy()

    for contour_data in contours_with_corners:
        top_left = contour_data['top_left']
        bottom_right = contour_data['bottom_right']
        
        # Draw the rectangles for valid contours
        cv2.rectangle(
            filtered_image,
            tuple(top_left),
            tuple(bottom_right),
            (0, 255, 0),  # Green color for valid rectangles
            2
        )
    
    return filtered_image

def remove_nested_rectangles(contours_with_corners):
    # Get a list of rectangle areas and their indices
    rectangle_areas = [(cv2.contourArea(data['contour']), data) for data in contours_with_corners]

    # Sort by area
    sorted_by_area = sorted(rectangle_areas, key=lambda x: x[0], reverse=True)

    filtered_contours = []
    for i, data in enumerate(sorted_by_area):
        is_nested = False
        for j in range(i + 1, len(sorted_by_area)):
            if sorted_by_area[j][0] < data[0] * 0.5:  # Check if inner rectangle is much smaller
                is_nested = True
                break

        if not is_nested:
            filtered_contours.append(data[1])

    return filtered_contours

def find_similar_width_and_aligned_edges(matched_pairs, width_tolerance=20, x_axis_tolerance=10):
    aligned_contours = []

    for contour_pair in matched_pairs:
        contour1 = contour_pair[0]['contour']
        contour2 = contour_pair[1]['contour']

        # Get bounding rectangles for the contours
        x1, y1, w1, h1 = cv2.boundingRect(contour1)
        x2, y2, w2, h2 = cv2.boundingRect(contour2)

        # Calculate the difference in widths
        width_difference = abs(w1 - w2)

        # Check if the left edges are aligned within tolerance
        left_edge_difference = abs(x1 - x2)

        # Check if the right edges are aligned within tolerance
        right_edge_difference = abs((x1 + w1) - (x2 + w2))

        # If width difference and edge alignment are within tolerances
        if (width_difference <= width_tolerance and 
            left_edge_difference <= x_axis_tolerance and
            right_edge_difference <= x_axis_tolerance):
            aligned_contours.append((contour_pair[0], contour_pair[1]))  # Add to aligned pairs

    return aligned_contours


# Define your target colors and tolerances
target_colors_bgr = [
    [230, 234, 205],  # Light green
    [190, 200, 160],  # Dark green
    [219, 215, 247]   # Red
]

tolerances = [  # White background
    np.array([20, 20, 20]),  # Tolerance for the first color
    np.array([20, 20, 20]),
    np.array([15, 15, 15])   # Tolerance for the third color
]

# Process and classify the image


def process_and_classify_image(image_path):
    base_name = os.path.basename(image_path)
    file_name, _ = os.path.splitext(base_name)

    target_image = cv2.imread(image_path)
    if target_image is None:
        raise ValueError("Invalid image path or unable to read image.")

    display_image(target_image, 'Original Image',
                  save_image=False, save_path=None, filename=None)
    preprocessed_image = preprocess_image(target_image)

    # Code for black background
    background = check_background_color(target_image)
    print(background)
    if background == 'Black':
        hsv_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2HSV)

        # Define HSV color ranges for green and red
        # Adjust the lower and upper bounds as needed to capture the colors accurately
        green_lower = np.array([35, 50, 50])  # Lower bound for green
        green_upper = np.array([95, 255, 255])  # Upper bound for green
        red_lower1 = np.array([0, 50, 50])  # Lower bound for red
        red_upper1 = np.array([5, 255, 255])  # Upper bound for red
        # Lower bound for red (for reds near 0)
        red_lower2 = np.array([170, 50, 50])
        # Upper bound for red (for reds near 180)
        red_upper2 = np.array([175, 255, 255])

        # Create masks for green and red
        mask_green = cv2.inRange(hsv_image, green_lower, green_upper)
        mask_red1 = cv2.inRange(hsv_image, red_lower1, red_upper1)
        mask_red2 = cv2.inRange(hsv_image, red_lower2, red_upper2)

        # Combine the red masks to get all red regions
        mask_red = cv2.bitwise_or(mask_red1, mask_red2)

        kernel = np.ones((5, 5), np.uint8)
        cleaned_mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
        cleaned_mask_red = cv2.morphologyEx(
            cleaned_mask_red, cv2.MORPH_CLOSE, kernel)
        cleaned_mask_green = cv2.morphologyEx(
            mask_green, cv2.MORPH_OPEN, kernel)
        cleaned_mask_green = cv2.morphologyEx(
            cleaned_mask_green, cv2.MORPH_CLOSE, kernel)

        # display_mask(mask_red, 'Red Mask')
        # display_mask(mask_green, 'Green Mask')
        contours_red, _ = cv2.findContours(
            cleaned_mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_green, _ = cv2.findContours(
            cleaned_mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # List to store contour information
        contour_data = []

        # Loop through red contours to get bounding rectangles and the number of edges
        for contour in contours_red:
            x, y, w, h = cv2.boundingRect(contour)
            # Approximate contour to find the number of edges
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            contour_data.append({
                'color': 'Red',
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'num_edges': len(approx)
            })

        # Loop through green contours to get bounding rectangles and the number of edges
        for contour in contours_green:
            x, y, w, h = cv2.boundingRect(contour)
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            contour_data.append({
                'color': 'Green',
                'x': x,
                'y': y,
                'width': w,
                'height': h,
                'num_edges': len(approx)
            })

        # Print out the contour data in a table
        # print("Color | X | Y | Width | Height | Num. Edges")
        # print("-------------------------------------------")
        # for data in contour_data:
        #     print(f"{data['color']}  | {data['x']}  | {data['y']}  | {data['width']}  | {data['height']}  | {data['num_edges']}")

        temp_image = np.zeros_like(target_image)

        # Draw all red contours
        cv2.drawContours(temp_image, contours_red, -1, (255, 0, 0), 2)

        # Draw all green contours
        cv2.drawContours(temp_image, contours_green, -1, (0, 255, 0), 2)

        # Display the visualization
        display_image(temp_image, "All Contours Before Filtering", save_image=False, save_path=None, filename=None)

        image_width = target_image.shape[1]

        # Define the right-side threshold (last 5% of the image width)
        right_threshold = image_width * 0.95  # 95% of the image width

        # Filter out contours located in the last 5% on the right
        filtered_contours_red = [
            contour for contour in contours_red
            if cv2.boundingRect(contour)[0] < right_threshold
        ]

        filtered_contours_green = [
            contour for contour in contours_green
            if cv2.boundingRect(contour)[0] < right_threshold
        ]

        image_height = target_image.shape[0]
        top_threshold = image_height * 0.1  # 5% of the image height

        # Filter out contours located in the top 5%
        filtered_contours_red = [
            contour for contour in filtered_contours_red
            if cv2.boundingRect(contour)[1] > top_threshold
        ]

        filtered_contours_green = [
            contour for contour in filtered_contours_green
            if cv2.boundingRect(contour)[1] > top_threshold
        ]

        max_width_threshold = 500

        # Apply the width filter to red contours
        filtered_contours_red = [
            contour for contour in filtered_contours_red
            if cv2.boundingRect(contour)[1] > top_threshold
            and cv2.boundingRect(contour)[2] <= max_width_threshold  # Width check
        ]

        # Apply the width filter to green contours
        filtered_contours_green = [
            contour for contour in filtered_contours_green
            if cv2.boundingRect(contour)[1] > top_threshold
            and cv2.boundingRect(contour)[2] <= max_width_threshold  # Width check
        ]

        filtered_contours_red = clip_contours_to_right(
            filtered_contours_red, image_width, right_ratio=0.6)
        filtered_contours_green = clip_contours_to_right(
            filtered_contours_green, image_width, right_ratio=0.6)

   
        filtered_contours_red = filter_by_edge_count(
            filtered_contours_red, max_edges=20)
        filtered_contours_green = filter_by_edge_count(
            filtered_contours_green, max_edges=20)

        min_area = 50  # Adjust this value as needed
        filtered_contours_red = filter_by_minimum_area(
            filtered_contours_red, min_area)
        filtered_contours_green = filter_by_minimum_area(
            filtered_contours_green, min_area)

        combined_contour_image = np.zeros_like(target_image)
        cv2.drawContours(combined_contour_image,
                         filtered_contours_red, -1, (0, 0, 255), 1)

        # Draw green contours
        cv2.drawContours(combined_contour_image,
                         filtered_contours_green, -1, (0, 255, 0), 1)

        # Display the combined contour image
        display_image(combined_contour_image, 'Combined Red and Green Contours',
                      save_image=False, save_path=None, filename=None)
        # Combine red and green contours for processing

        all_contours = filtered_contours_red + filtered_contours_green
        simplified_contours = [simplify_contour(contour) for contour in all_contours]

        # List to store contour information with centers and additional data
        contours_with_centers = []

        # Loop through each contour with its index
        for index, contour in enumerate(simplified_contours):
            # Get the bounding rectangle for the contour
            x, y, w, h = cv2.boundingRect(contour)

            top_left = (x, y)
            top_right = (x + w, y)
            bottom_left = (x, y + h)
            bottom_right = (x + w, y + h)

            contour_area = cv2.contourArea(contour)

            # The uppermost horizontal edge's center point (x-coordinate)
            uppermost_horizontal_center_x = x + (w / 2)

            # The y-coordinate of the top edge (uppermost horizontal edge)
            top_edge_y = y

            # Calculate the width
            contour_width = w

            leftmost_x = x

            # Add the calculated data to the list with the contour number
            contours_with_centers.append({
                "contour": contour,
                "contour_number": index + 1,  # Numbering starting from 1
                "center_x": uppermost_horizontal_center_x,
                "top_edge_y": y,
                "width": contour_width,
                "leftmost_x": leftmost_x,
                "top_left": top_left,
                "top_right": top_right,
                "bottom_left": bottom_left,
                "bottom_right": bottom_right,
                "area": contour_area
            })

        # Display the results in a simple tabular format
        print("Contour | Center X | Top Edge Y | Width | Leftmost X | Top Left | Top Right | Bottom Left | Bottom Right | Area")
        print("---------------------------------------")
        for contour_data in contours_with_centers:
            top_left = contour_data['top_left']
            top_right = contour_data['top_right']
            bottom_left = contour_data['bottom_left']
            bottom_right = contour_data['bottom_right']
            print(
                f"{contour_data['contour_number']}     | "
                f"{contour_data['center_x']:.2f}     | "
                f"{contour_data['top_edge_y']}     | "
                f"{contour_data['width']:.2f}    |" 
                f"{contour_data['leftmost_x']:.2f}     | "
                f"{top_left}     | "
                f"{top_right}     | "
                f"{bottom_left}     | "
                f"{contour_data['bottom_right']}     | "
                f"{contour_data['area']:.2f}"
            )

        # Create a new image to draw contours on
        image_with_contour_numbers = np.zeros_like(target_image)

        # Draw the contours on the new image
        cv2.drawContours(
            image_with_contour_numbers,
            [data['contour'] for data in contours_with_centers],  # Extract the contours from the dictionary
            -1,
            (255, 255, 255),  # White color for the contours
            2
        )

        # Loop through the contours and add text with the contour number and center point
        for contour_data in contours_with_centers:
            contour_number = contour_data["contour_number"]
            center_x = contour_data["center_x"]
            top_edge_y = contour_data["top_edge_y"]

            # Draw the contour number text at the uppermost horizontal edge
            text_position = (int(center_x), int(top_edge_y - 10))  # Slightly above the edge
            cv2.putText(
                image_with_contour_numbers,
                f"{contour_number}",
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),  # Green text
                2,
            )
            # print('Contour Numbers')
            # print("Contour | Center X | Top Edge Y | Width")
            # print(
            #     f"{contour_data['contour_number']}     | "
            #     f"{contour_data['center_x']:.2f}     | "
            #     f"{contour_data['top_edge_y']}     | "
            #     f"{contour_data['width']:.2f}"
            # )

        # Display the image with contour numbers
        display_image(
            image_with_contour_numbers,
            'Contour Numbers',
            save_image=False,  # Set to True to save the image
            save_path=None,  # Provide a save path if needed
            filename=None  # Provide a filename if saving
        )

        # Example code to draw valid rectangles
        image_with_rectangles = draw_rectangles(target_image, contours_with_centers)
        # valid_contours_with_corners = remove_nested_rectangles(valid_contours_with_corners)
        valid_contours_with_corners = filter_rectangular_contours(contours_with_centers, minimum_fill_ratio=0.5)
        image_with_filtered_rectangles = draw_filtered_rectangles(target_image, valid_contours_with_corners)
        # Display the updated image
        display_image(image_with_rectangles, 'Validated Rectangles')
        display_image(image_with_filtered_rectangles, 'Valid Filtered Rectangles')

        # Display the contour number for the most rectangular contour
        print("Most Rectangular Contour Number:", find_most_rectangular(valid_contours_with_corners))

        if len(valid_contours_with_corners) == 2:
            # If there are only two contours, they are the matched pair
            matched_pairs = [(valid_contours_with_corners[0], valid_contours_with_corners[1])]
        else:
            # If there are more than two, use the matching function
            matched_pairs = match_contour_pairs(
                valid_contours_with_corners, x_distance_threshold=15, width_tolerance=100
            )

        # Find matched contour pairs
        # matched_pairs = match_contour_pairs(contours_with_centers, x_distance_threshold=15, width_tolerance=100)
        # print("Matched pairs:", matched_pairs)

        # Create an image to draw contours
        # Get the contours directly from contours_with_centers
        # all_contours = [data['contour'] for data in contours_with_centers]

        # Function to get contour by contour number from contours_with_centers
        def get_contour_by_number(contour_number, contours_with_centers):
            return contours_with_centers[contour_number - 1]['contour']

        aligned_contours = find_similar_width_and_aligned_edges(matched_pairs, width_tolerance=50, x_axis_tolerance=50)
        final_contours_image = np.zeros_like(target_image)
        
        if aligned_contours:
            print("Aligned Contours:")
            for contour1, contour2 in aligned_contours:
                contour1_number = contour1['contour_number']
                contour2_number = contour2['contour_number']
                print(f"Contour {contour1_number} is aligned with Contour {contour2_number}")

            # Visualize the aligned contours
            aligned_contours_image = np.zeros_like(final_contours_image)
            for contour1, contour2 in aligned_contours:
                cv2.drawContours(aligned_contours_image, [contour1['contour']], -1, (255, 0, 0), 3)  # Red
                cv2.drawContours(aligned_contours_image, [contour2['contour']], -1, (0, 255, 0), 3)  # Green

            display_image(
                aligned_contours_image,
                'Aligned Contours',
                save_image=False,
                save_path=None,
                filename=None
            )
        else:
            print("No aligned contours found.")
        # Create a new image to draw the matched pairs on
        image_with_matched_pairs = np.zeros_like(target_image)  # Use a similar size to the target image

        # Loop through the matched pairs and draw the contours
        for pair in aligned_contours:
            # Get the contours from the pair
            contour1 = get_contour_by_number(pair[0]['contour_number'], contours_with_centers)
            contour2 = get_contour_by_number(pair[1]['contour_number'], contours_with_centers)

            # Determine which contour is higher on the Y-axis
            if pair[0]["top_edge_y"] < pair[1]["top_edge_y"]:
                higher_contour = contour1
                lower_contour = contour2
            else:
                higher_contour = contour2
                lower_contour = contour1

            # Draw the higher contour in green
            cv2.drawContours(image_with_matched_pairs, [higher_contour], -1, (0, 255, 0), 2)  # Green
            
            # Draw the lower contour in red
            cv2.drawContours(image_with_matched_pairs, [lower_contour], -1, (255, 0, 0), 2)  # Red
            
            # Add contour numbers to the image
            cv2.putText(
                image_with_matched_pairs,
                f"{pair[0]['contour_number']}",
                (int(pair[0]['center_x']), int(pair[0]['top_edge_y'] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            
            cv2.putText(
                image_with_matched_pairs,
                f"{pair[1]['contour_number']}",
                (int(pair[1]['center_x']), int(pair[1]['top_edge_y'] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        # Display the image with matched pairs and their contour numbers
        display_image(
            image_with_matched_pairs,
            'Matched Contours',
            save_image=False,
            save_path=None,
            filename=None,
        )

        # Create a new image to draw the final contours
        final_contours_image = np.zeros_like(target_image)

        # Draw the filtered contours with appropriate colors
        for index, contour_pair in enumerate(aligned_contours):
            # Determine which contour is higher on the Y-axis
            if contour_pair[0]['top_edge_y'] < contour_pair[1]['top_edge_y']:
                lower_contour = contour_pair[0]['contour']
                higher_contour = contour_pair[1]['contour']
            else:
                lower_contour = contour_pair[1]['contour']
                higher_contour = contour_pair[0]['contour']

            # Draw the lower contour in red
            cv2.drawContours(final_contours_image, [lower_contour], -1, (255, 0, 0), 3)  # Red

            # Draw the higher contour in green
            cv2.drawContours(final_contours_image, [higher_contour], -1, (0, 255, 0), 3)  # Green

        # Display the final contours image
        display_image(
            final_contours_image,
            'Final Contours',
            save_image=False,  # Set to True to save the image
            save_path=None,
            filename=None,
        )

        contour_areas = [(cv2.contourArea(contour_data['contour']), contour_data) for contour_pair in aligned_contours for contour_data in contour_pair]

        if contour_areas:
            # Sort the list by area in descending order
            contour_areas_sorted = sorted(contour_areas, key=lambda x: x[0], reverse=True)

            # Select the top two contours based on area
            top_two_contours = [data[1] for data in contour_areas_sorted[:2]]

            matched_pairs = [(top_two_contours[0], top_two_contours[1])]

        else:
            # If no contour areas, matched_pairs is empty, and we need to initialize the output variable
            matched_pairs = []
            top_two_contours = []

        shaded_contours_image = np.zeros_like(target_image)
        if matched_pairs:
            # Draw and fill the contours with their original colors on the black background
            for contour_pair in matched_pairs:
                # Draw each contour in the pair
                for contour_data in contour_pair:
                    contour = contour_data['contour']
                    
                    # Create a mask for the current contour
                    contour_mask = np.zeros_like(mask_red)
                    cv2.drawContours(contour_mask, [contour], -1, 255, -1)

                    # Compute the mean color of the contour area in the original image
                    mean_val = cv2.mean(target_image, mask=contour_mask)
                    mean_color = (int(mean_val[0]), int(mean_val[1]), int(mean_val[2]))

                    # Fill the contour with the mean color on the black background image
                    cv2.drawContours(shaded_contours_image, [contour], -1, mean_color, cv2.FILLED)

            # Display the shaded contours image with black background
            display_image(
                shaded_contours_image,
                'Shaded Contours with Black Background',
                save_image=False,  # Set to True to save the image
                save_path='test_images',
                filename='Test24b'
            )

        if matched_pairs:
            desired_contours = [contour_data['contour'] for pair in matched_pairs for contour_data in pair]
            position = classify_long_short_or_unknown_black_background(desired_contours, shaded_contours_image)
        else:
            # Handle the case where there are no matched pairs
            desired_contours = []
            position = 'No Matched Pairs'

        # desired_contours = [contour_data['contour'] for pair in matched_pairs for contour_data in pair]

        # Classify whether it's long or short using these contours and the shaded image with the black background

        print(position)
        return position
    else:
        # Find colors
        masks = find_colors(preprocessed_image, target_colors_bgr, tolerances)

        cleaned_mask_green, cleaned_mask_red = clean_and_combine_masks(masks)
        # Display the cleaned masks
        display_mask(cleaned_mask_red, 'Red Mask')
        display_mask(cleaned_mask_green, 'Green Mask')
        contours_red, _ = cv2.findContours(
            cleaned_mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_green, _ = cv2.findContours(
            cleaned_mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        all_contours = contours_red + contours_green
        # Classify as long or short
        position = classify_long_short_or_unknown_black_background(
            all_contours, preprocessed_image)

        # Create folder and save image
        save_path = create_save_path("saved_images", position)
        sanitized_filename = sanitize_filename(file_name)

        save_file_path = os.path.join(
            save_path, f"{sanitized_filename}_classified.png")
        cv2.imwrite(save_file_path, preprocessed_image)

        print(f"Image saved as {save_file_path}")

        return position


# Command-line interface
if __name__ == "__main__":
    # Use argparse to parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Process an image to classify it as 'long', 'short', or 'unknown'.")
    parser.add_argument("image_path", type=str,
                        help="Path to the image to be processed.")

    # Parse the arguments
    args = parser.parse_args()

    # Process and classify the image based on the provided image path
    result = process_and_classify_image(args.image_path)

    print(result)  # Output the result to the console
