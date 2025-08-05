"""Real-time Dice Recognition System using OpenCV.

This module implements a real-time dice recognition system using OpenCV. It can detect
and count dots on dice faces using blob detection, supporting both light and dark dots.
The system processes webcam input and provides live visualization of detected dice and their values.

Dependencies:
    - OpenCV (cv2): For image processing and computer vision tasks
    - NumPy: For numerical operations and array handling
    - scikit-learn: For clustering dots into dice groups
    - scipy: For spatial distance calculations and geometric operations

Author: Louis Watermeyer
"""

# External dependencies
import cv2
import numpy as np
from sklearn import cluster
from typing import List

# Internal dependencies
from internal_data_classes import Blob, Dice


# Color constants for blob visualization
DARK_BLOB_COLOR = (255, 0, 0)  # Blue for dark blobs
LIGHT_BLOB_COLOR = (0, 0, 255)  # Red for light blobs

# Flag for enabling debug drawings
DEBUG_DRAWINGS = False

# Flag to select the correct camera
IS_WINDOWS = True

# Camera selection with example values
# The index or device path may need to be adjusted based on the system
WINDOWS_CAMERA_INDEX = 0
LINUX_CAMERA_PATH = "/dev/video4"


def create_blob_detector(is_dark=True) -> cv2.SimpleBlobDetector:
    """Create a blob detector configured for detecting dice dots.

    This function creates and configures a SimpleBlobDetector with parameters optimized
    for detecting dots on dice. It can detect both dark dots on light dice and
    light dots on dark dice depending on the input parameter.

    Args:
        is_dark (bool): If True, detects dark dots on light background.
                        If False, detects light dots on dark background.

    Returns:
        cv2.SimpleBlobDetector: Configured blob detector instance.
    """
    params = cv2.SimpleBlobDetector_Params()

    # Configure area filtering to detect reasonable dot sizes
    params.filterByArea = True
    params.minArea = 20  # Minimum dot size in pixels
    params.maxArea = 400  # Maximum dot size in pixels

    # Ensure detected blobs are roughly circular
    params.filterByCircularity = True
    params.minCircularity = 0.6  # Perfect circle = 1.0

    # Filter by convexity to ensure solid, well-formed dots
    params.filterByConvexity = True
    params.minConvexity = 0.6  # Perfect convex shape = 1.0

    # Filter by inertia ratio to detect round-ish shapes
    params.filterByInertia = True
    params.minInertiaRatio = 0.5  # Perfect circle = 1.0

    # Configure color filtering based on dot type
    params.filterByColor = True
    # 0 for dark blobs, 255 for light blobs
    params.blobColor = 0 if is_dark else 255

    return cv2.SimpleBlobDetector_create(params)


def get_blobs_from_frame(frame: np.ndarray) -> List["Blob"]:
    """Detect dots (blobs) in the input frame.

    This function processes the input frame to detect both dark and light dots
    using blob detectors. It includes preprocessing steps to improve
    detection accuracy and returns a list of Blob objects.

    The process follows these steps:
    1. Detector Creation: Create specialized blob detectors for dark and light dots
    2. Image Preprocessing: Convert to grayscale and apply noise reduction
    3. Blob Detection: Detect both dark and light blobs using blob detectors
    5. Visualization: (Optional) Display detected blobs for debugging

    Args:
        frame (np.ndarray): Input BGR image frame from the video feed.

    Returns:
        list[Blob]: List of detected blobs.
    """
    # ===== STEP 1: DETECTOR CREATION =====
    # Create specialized blob detectors for different dot types
    detector_dark = create_blob_detector(
        is_dark=True)  # For dark dots on light dice
    detector_light = create_blob_detector(
        is_dark=False)  # For light dots on dark dice

    # ===== STEP 2: IMAGE PREPROCESSING =====
    # Convert to grayscale for more effective blob detection
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter to reduce noise while preserving edges
    # Parameters: source, diameter of pixel neighborhood, color sigma, space sigma
    #  - Diameter=9: Size of pixel neighborhood for filtering
    #  - Color sigma=75: Higher values mix colors more
    #  - Space sigma=75: Higher values mix farther pixels
    filtered_frame = cv2.bilateralFilter(grayscale_frame, 9, 75, 75)

    # ===== STEP 3: BLOB DETECTION =====
    detected_blobs = []

    # Convert dark keypoints to Blob objects and add to results
    for dark_keypoint in detector_dark.detect(filtered_frame):
        detected_blobs.append(Blob.from_keypoint(dark_keypoint, is_dark=True))

    # Convert light keypoints to Blob objects and add to results
    for light_keypoint in detector_light.detect(filtered_frame):
        detected_blobs.append(Blob.from_keypoint(
            light_keypoint, is_dark=False))

    # ===== STEP 5: VISUALIZATION (OPTIONAL) =====
    # If debug mode is enabled, create a visual representation of detected blobs
    if DEBUG_DRAWINGS:
        # Create a copy of the original frame for visualization
        visualization_frame = frame.copy()

        # Draw each blob as a circle with appropriate color
        for blob in detected_blobs:
            blob_position = blob.position
            if blob_position is not None:
                # Calculate circle radius from blob size
                blob_radius = blob.size / 2
                # Choose color based on blob type (blue for dark, red for light)
                blob_color = DARK_BLOB_COLOR if blob.is_dark else LIGHT_BLOB_COLOR

                # Draw the circle at the blob's position
                cv2.circle(
                    visualization_frame,
                    (int(blob_position[0]), int(blob_position[1])),
                    int(blob_radius),
                    blob_color,
                    2,  # Line thickness
                )

        # Display the visualization with all detected blobs
        cv2.imshow("All detected dots", visualization_frame)

    # Return the complete list of detected blobs for further processing
    return detected_blobs


def check_3_blobs_in_line(blobs: List["Blob"]) -> bool:
    """Validate if 3 blobs form a straight line pattern.

    A valid line pattern has:
    - One blob in the center
    - Two outer blobs at approximately equal distances from the center
    - The two outer blobs at approximately 180° from each other relative to center

    Args:
        blobs (List["Blob"]): List of exactly 3 Blob objects to check

    Returns:
        bool: True if blobs form a line pattern, False otherwise
    """
    # Require exactly 3 blobs for a line pattern
    if len(blobs) != 3:
        return False

    # ===== STEP 1: FIND CENTER BLOB =====
    # Calculate geometric center (centroid) of all blobs
    blob_positions = [blob.position for blob in blobs]
    centroid = np.mean(blob_positions, axis=0)

    # Find distances from each blob to the centroid
    distances_to_centroid = np.sqrt(
        np.sum((np.array(blob_positions) - centroid) ** 2, axis=1)
    )

    # The blob closest to the centroid is likely the center blob
    center_blob_index = np.argmin(distances_to_centroid)
    center_position = blob_positions[center_blob_index]

    # ===== STEP 2: VALIDATE OUTER BLOBS =====
    # Get indices of the two outer blobs
    outer_blob_indices = [i for i in range(3) if i != center_blob_index]
    if len(outer_blob_indices) != 2:
        return False

    # Calculate distances from center blob to each outer blob
    blob_positions_array = np.array(blob_positions)
    distance_to_first_outer = np.sqrt(
        np.sum(
            (blob_positions_array[outer_blob_indices[0]] - center_position) ** 2)
    )
    distance_to_second_outer = np.sqrt(
        np.sum(
            (blob_positions_array[outer_blob_indices[1]] - center_position) ** 2)
    )

    # Prevent division by zero
    if distance_to_first_outer < 1 or distance_to_second_outer < 1:
        return False  # If any distance is too small, it's not a valid line

    # Check if distances are approximately equal (within 25% tolerance)
    # This allows for some perspective distortion
    distance_ratio = distance_to_first_outer / distance_to_second_outer
    if not (0.75 <= distance_ratio <= 1.25):
        return False

    # ===== STEP 3: VALIDATE ANGLE =====
    # Calculate angles from center to each outer blob
    angle_to_first_outer = np.arctan2(
        blob_positions_array[outer_blob_indices[0]][1] - center_position[1],
        blob_positions_array[outer_blob_indices[0]][0] - center_position[0],
    )
    angle_to_second_outer = np.arctan2(
        blob_positions_array[outer_blob_indices[1]][1] - center_position[1],
        blob_positions_array[outer_blob_indices[1]][0] - center_position[0],
    )

    # Normalize angles to [0, 2π] range
    angle_to_first_outer = (angle_to_first_outer + 2 * np.pi) % (2 * np.pi)
    angle_to_second_outer = (angle_to_second_outer + 2 * np.pi) % (2 * np.pi)

    # Calculate the absolute angle difference
    angle_difference = abs(angle_to_first_outer - angle_to_second_outer)

    # For a straight line, the angle difference should be approximately 180° (π radians)
    # Allow ±20° tolerance (π/9 radians)
    return abs(angle_difference - np.pi) <= np.pi / 9


def check_square_formation(blobs: List["Blob"]) -> bool:
    """Validate if 4 blobs form a square pattern.

    A valid square pattern has specific distance relationships:
    - Each dot has two neighbors at equal distances (square sides)
    - Each dot has one neighbor at distance sqrt(2) times the side length (diagonal)

    Args:
        blobs (List["Blob"]): List of exactly 4 Blob objects to check

    Returns:
        bool: True if blobs form a square pattern, False otherwise
    """
    # Require exactly 4 blobs for a square pattern
    if len(blobs) != 4:
        return False

    # Tools to calculate pairwise distances between all dots
    from scipy.spatial.distance import pdist, squareform
    import math

    # Extract blob positions and compute distance matrix
    blob_positions = [blob.position for blob in blobs]
    distance_matrix = squareform(pdist(blob_positions))

    # For a square, each dot must have specific distance relationships to others
    valid_corner_count = 0

    # Check each dot's distance relationships
    for corner_index in range(4):
        # Get distances from this dot to all others, sorted
        distances_to_other_corners = sorted(distance_matrix[corner_index])

        # Skip the first distance (distance to itself = 0)
        distances_to_other_corners = distances_to_other_corners[1:]

        # In a square pattern, we expect:
        if len(distances_to_other_corners) == 3:  # Should have 3 neighbors
            # The first two distances should be approximately equal (square sides)
            if distances_to_other_corners[0] > 0:  # Prevent division by zero
                side_ratio = (
                    distances_to_other_corners[1] /
                    distances_to_other_corners[0]
                )

                # The third distance should be approximately sqrt(2) times the side length (diagonal)
                diagonal_ratio = (
                    distances_to_other_corners[2] /
                    distances_to_other_corners[0]
                )
                expected_diagonal_ratio = math.sqrt(2)

                # Allow 20% tolerance in the ratios to account for perspective distortion
                is_valid_side_ratio = 0.8 <= side_ratio <= 1.2
                is_valid_diagonal_ratio = (
                    expected_diagonal_ratio * 0.8
                    <= diagonal_ratio
                    <= expected_diagonal_ratio * 1.2
                )

                if is_valid_side_ratio and is_valid_diagonal_ratio:
                    valid_corner_count += 1

    # All four corners must satisfy the distance criteria
    return valid_corner_count == 4


def check_5_dot_pattern(blobs: List["Blob"]) -> bool:
    """Validate if 5 blobs form a valid dice pattern with 4 corner dots and 1 center dot.

    A valid 5-dot pattern has specific geometric characteristics:
    - Four dots should form a square pattern at the corners
    - One dot should be positioned at the center of the square

    The process follows these steps:
    1. Centroid Calculation: Find the geometric center of all dots combined
    2. Center Dot Identification: Locate the dot closest to the centroid
    3. Corner Dots Extraction: Identify the four dots forming the corners
    4. Square Validation: Verify the corner dots form a square
    5. Center Position Verification: Confirm center dot is inside the square

    Args:
        blobs (List["Blob"]): List of exactly 5 Blob objects to check

    Returns:
        bool: True if blobs form a valid 5-dot dice pattern, False otherwise
    """
    # ===== STEP 1: CENTROID CALCULATION =====
    # Calculate the geometric center point of all blobs
    centroid_x_coordinate = sum(
        blob.position[0] for blob in blobs) / len(blobs)
    centroid_y_coordinate = sum(
        blob.position[1] for blob in blobs) / len(blobs)

    # ===== STEP 2: CENTER DOT IDENTIFICATION =====
    # Calculate distances from each blob to the centroid
    distances_to_centroid = [
        np.sqrt(
            (blob.position[0] - centroid_x_coordinate) ** 2
            + (blob.position[1] - centroid_y_coordinate) ** 2
        )
        for blob in blobs
    ]

    # Find the blob with minimum distance to centroid (center dot)
    center_dot_index = np.argmin(distances_to_centroid)
    center_dot = blobs[center_dot_index]

    # ===== STEP 3: FIND CORNER DOTS =====
    # Extract the four corner dots (all blobs except the center one)
    corner_dots = [blob for i, blob in enumerate(
        blobs) if i != center_dot_index]

    # ===== STEP 4: SQUARE VALIDATION =====
    # Verify the four corner dots form a square pattern
    if not check_square_formation(corner_dots):
        return False

    # ===== STEP 5: CENTER POSITION VERIFICATION =====
    # Calculate the bounding box of the corner dots
    min_x_coordinate = min(blob.position[0] for blob in corner_dots)
    max_x_coordinate = max(blob.position[0] for blob in corner_dots)
    min_y_coordinate = min(blob.position[1] for blob in corner_dots)
    max_y_coordinate = max(blob.position[1] for blob in corner_dots)

    # Check if center dot is inside the square boundary
    is_center_inside_square = (
        min_x_coordinate <= center_dot.position[0] <= max_x_coordinate
        and min_y_coordinate <= center_dot.position[1] <= max_y_coordinate
    )

    return is_center_inside_square


def check_6_dot_pattern(blobs: List["Blob"]) -> bool:
    """Validate if 6 blobs form a valid dice pattern with two parallel lines of 3 dots each.

    A valid 6-dot pattern has specific geometric characteristics:
    - The 6 dots should be divisible into two groups of 3 dots each
    - Each group of 3 dots should form a straight line

    The process follows these steps:
    1. Group Generation: Generate all possible ways to split 6 dots into two groups of 3
    2. Group Validation: For each possible grouping, check if both groups form valid lines
    3. Pattern Verification: Return true if any valid grouping is found

    Args:
        blobs (List["Blob"]): List of exactly 6 Blob objects to check

    Returns:
        bool: True if blobs form a valid 6-dot dice pattern, False otherwise
    """
    # ===== STEP 1: GROUP GENERATION =====
    # Import the combinations function for generating all possible groupings
    from itertools import combinations

    # Create a list of indices representing the blobs
    blob_index_list = list(range(len(blobs)))

    # Generate all possible ways to select 3 indices from the 6 indices
    # This will create all possible ways to split 6 dots into two groups of 3
    possible_three_dot_groupings = list(combinations(blob_index_list, 3))

    # ===== STEP 2: GROUP VALIDATION =====
    # Examine each possible grouping of the 6 dots
    for first_group_indices in possible_three_dot_groupings:
        # Determine the complementary indices for the second group
        second_group_indices = [
            index for index in blob_index_list if index not in first_group_indices
        ]

        # Get the actual blob objects for both groups
        first_dot_group = [blobs[index] for index in first_group_indices]
        second_dot_group = [blobs[index] for index in second_group_indices]

        # ===== STEP 3: PATTERN VERIFICATION =====
        # Check if both groups form valid lines
        if check_3_blobs_in_line(first_dot_group) and check_3_blobs_in_line(second_dot_group):
            return True

    # If no valid grouping is found, return False
    return False


def check_valid_dice_formation(blobs: List["Blob"]) -> bool:
    """Validate if a group of blobs forms a valid dice pattern.

    This function analyzes the spatial arrangement of dots (blobs) to determine
    if they match standard dice patterns. It checks different patterns based on
    the number of dots (1-6) using geometric properties like distances and angles.

    The validation process follows these steps:
    1. Initial validation: Check if the number of dots is valid for a die (1-6)
    2. Pattern recognition: Apply specific geometric tests based on dot count
       - For 1-2 dots: Accept with minimal validation
       - For 3 dots: Check if they form a straight line
       - For 4 dots: Verify square arrangement
       - For 5 dots: Verify four corner dots plus one center dot
       - For 6 dots: Verify two parallel lines of 3 dots each

    Args:
        blobs (List["Blob"]): List of Blob objects representing dots to be validated

    Returns:
        bool: True if the blobs form a valid dice pattern, False otherwise
    """
    # ===== STEP 1: INITIAL VALIDATION =====
    # Standard dice have 1-6 dots, reject any pattern outside this range
    if len(blobs) > 6 or len(blobs) < 1:
        return False

    # ===== STEP 2: PATTERN RECOGNITION BASED ON DOT COUNT =====
    # Standard dice patterns have 1-6 dots in specific arrangements:
    # - For 1: One dot in center
    # - For 2: Two dots in opposite corners
    # - For 3: One in center, two in opposite corners
    # - For 4: Four dots in four corners
    # - For 5: Four dots in corners, one in center
    # - For 6: Six dots, two rows of three dots each

    # ===== CASE 1-2: SIMPLE PATTERNS =====
    if len(blobs) <= 2:
        # For 1 and 2 dots, accept with minimal validation because ther eis no geometric
        # way of validating them
        return True

    # ===== CASE 3: THREE DOTS IN LINE =====
    elif len(blobs) == 3:
        # For 3 blobs, check if they form a straight line
        return check_3_blobs_in_line(blobs)

    # ===== CASE 4: SQUARE PATTERN =====
    elif len(blobs) == 4:
        # For 4 dots, verify they form a square arrangement
        return check_square_formation(blobs)

    # ===== CASE 5: FOUR CORNERS PLUS CENTER =====
    elif len(blobs) == 5:
        # For 5 dots, verify four corner dots plus one center dot
        return check_5_dot_pattern(blobs)

    # ===== CASE 6: TWO PARALLEL LINES =====
    elif len(blobs) == 6:
        # For 6 dots, verify they form two parallel lines of 3 dots each
        return check_6_dot_pattern(blobs)

    # If we reach here, the pattern doesn't match any valid dice configuration
    return False


def process_blob_group(
    blob_group: List["Blob"]
) -> List["Dice"]:
    """Process a group of same-colored blobs into dice.

    This function handles the clustering of blobs into dice and the recursive refinement
    for complex cases where dice are close together.

    The processing follows these steps:
    1. Initial Clustering: Group nearby blobs using DBSCAN with loose parameters
    2. Pattern Validation: Check if each cluster forms a valid dice pattern
    3. Recursive Refinement: For invalid clusters, try progressively tighter clustering
       to separate potentially overlapping dice

    Args:
        blob_group (List["Blob"]): List of blobs of the same color (dark or light)

    Returns:
        List["Dice"]: List of detected dice objects from this blob group
    """
    # Skip processing if no blobs in this group
    if not blob_group:
        return []  # No blobs to process, return empty list

    detected_group_dice = []

    # ===== STEP 1: INITIAL CLUSTERING =====
    # Extract blob positions for clustering
    blob_positions = np.array([blob.position for blob in blob_group])

    # Ensure positions array is properly shaped for DBSCAN
    # DBSCAN requires a 2D array even if there's only one point
    if len(blob_positions) == 1:
        blob_positions = blob_positions.reshape(1, -1)

    # Perform initial clustering with loose parameters
    # eps=40: Blobs within 40 pixels of each other are considered part of the same die
    # min_samples=1: Each blob can form its own cluster if isolated
    initial_clustering = cluster.DBSCAN(
        eps=40, min_samples=1).fit(blob_positions)

    # ===== STEP 2: PATTERN VALIDATION =====
    # For each cluster identified by DBSCAN
    position_to_blob_map = {tuple(blob.position): blob for blob in blob_group}
    for cluster_index in range(max(initial_clustering.labels_) + 1):
        # Get positions of all blobs in this cluster
        cluster_positions = blob_positions[initial_clustering.labels_ == cluster_index]

        # Skip empty clusters
        if len(cluster_positions) > 0:
            # Convert positions back to blob objects
            clustered_blobs = [
                position_to_blob_map[tuple(position)] for position in cluster_positions
            ]

            # Check if the cluster forms a valid dice pattern
            if check_valid_dice_formation(clustered_blobs):
                # Valid dice pattern found - create a Dice object
                die_centroid = tuple(np.mean(cluster_positions, axis=0))
                detected_group_dice.append(
                    Dice(position=die_centroid, blobs=clustered_blobs)
                )
            else:
                # ===== STEP 3: RECURSIVE REFINEMENT FOR COMPLEX CLUSTERS =====
                # If the cluster doesn't form a valid dice pattern, it might contain
                # multiple dice that are close together. Try to split it with tighter parameters.

                # Define a series of progressively tighter clustering parameters
                # Start loose and get tighter to gradually separate dice
                refinement_distance_thresholds = [
                    30,
                    25,
                    20,
                    15,
                    10,
                ]  # Progressively tighter clustering distances

                # Track which positions have been successfully assigned to dice
                processed_blob_positions = (
                    set()
                )  # Set of positions already processed into dice

                # Try each distance threshold in sequence
                for current_distance_threshold in refinement_distance_thresholds:
                    # Skip if we've already processed all positions in this cluster
                    if len(processed_blob_positions) == len(cluster_positions):
                        break

                    # Get positions that haven't been processed yet
                    remaining_blob_positions = np.array(
                        [
                            position
                            for position in cluster_positions
                            if tuple(position) not in processed_blob_positions
                        ]
                    )

                    # Skip if no positions left to process
                    if len(remaining_blob_positions) == 0:
                        break

                    # Ensure array is properly shaped for DBSCAN
                    if len(remaining_blob_positions) == 1:
                        remaining_blob_positions = remaining_blob_positions.reshape(
                            1, -1
                        )

                    # Try clustering with current tighter distance threshold
                    refinement_clustering = cluster.DBSCAN(
                        eps=current_distance_threshold, min_samples=1
                    ).fit(remaining_blob_positions)

                    # Process each sub-cluster from the refinement
                    valid_dice_found = False  # Flag to track if we found any valid dice

                    for refinement_cluster_index in range(
                        max(refinement_clustering.labels_) + 1
                    ):
                        # Get positions in this refinement sub-cluster
                        refinement_cluster_positions = remaining_blob_positions[
                            refinement_clustering.labels_ == refinement_cluster_index
                        ]

                        if len(refinement_cluster_positions) > 0:
                            # Convert positions back to blob objects
                            refinement_cluster_blobs = [
                                position_to_blob_map[tuple(position)]
                                for position in refinement_cluster_positions
                            ]

                            # Validate the refined cluster as a dice pattern
                            if check_valid_dice_formation(refinement_cluster_blobs):
                                # Valid dice found in the refinement - create a Dice object
                                refinement_centroid = tuple(
                                    np.mean(
                                        refinement_cluster_positions, axis=0)
                                )
                                detected_group_dice.append(
                                    Dice(
                                        position=refinement_centroid,
                                        blobs=refinement_cluster_blobs,
                                    )
                                )

                                # Mark these positions as processed
                                for position in refinement_cluster_positions:
                                    processed_blob_positions.add(
                                        tuple(position))

                                valid_dice_found = True

                    # Optimization - if we found valid dice at this distance threshold and
                    # there are still unprocessed positions, try the same threshold again
                    # before moving to a tighter value
                    if valid_dice_found and len(processed_blob_positions) < len(
                        cluster_positions
                    ):
                        # Insert the current threshold at the beginning to try again with the same value
                        refinement_distance_thresholds.insert(
                            0, current_distance_threshold
                        )

    # Return all detected dice from this blob group
    return detected_group_dice


def get_dice_from_blobs(blobs: List["Blob"], frame=None) -> List["Dice"]:
    """Group detected blobs into dice using spatial clustering.

    This function uses DBSCAN clustering to group nearby dots into dice based on
    their spatial proximity. It handles both dark and light blobs separately and
    validates the geometric patterns to ensure they match standard dice formations.

    The clustering process follows these steps:
    1. Process Color Groups: Apply the process_blob_group function to dark and light blobs
    2. Visualization: (Optional) Display detected dice centroids for debugging

    Args:
        blobs (List["Blob"]): List of detected blobs (dots)
        frame (np.ndarray, optional): Original image frame for visualization

    Returns:
        List["Dice"]: List of Dice objects, each containing its position and associated blobs
    """
    detected_dice = []

    # ===== STEP 1: PROCESS BLOBS BY COLOR =====
    # Separate blobs by color (dark vs light)
    # This prevents mixing dots from different colored dice
    dark_blobs = [blob for blob in blobs if blob.is_dark]
    light_blobs = [blob for blob in blobs if not blob.is_dark]

    # Process dark blobs (dark dots on light dice)
    dark_dice = process_blob_group(dark_blobs)

    # Process light blobs (light dots on dark dice)
    light_dice = process_blob_group(light_blobs)

    # Combine results from both color groups
    detected_dice = dark_dice + light_dice

    # ===== STEP 2: VISUALIZATION (OPTIONAL) =====
    # If debug mode is enabled and a frame is provided, visualize the dice centroids
    if DEBUG_DRAWINGS and frame is not None:
        visualization_frame = frame.copy()
        for die in detected_dice:
            # Draw a green circle at the centroid of each detected die
            cv2.circle(
                visualization_frame,
                (int(die.position[0]), int(die.position[1])),
                10,  # Radius of 10 pixels
                (0, 255, 0),  # Green color
                -1,  # Filled circle
            )
        # Display the visualization
        cv2.imshow("Detected Dice Centroids", visualization_frame)

    # Return the list of detected dice
    return detected_dice


def overlay_info(frame: np.ndarray, dice: List["Dice"]):
    """Overlay visual information about detected dice and dots on the frame.

    This function draws visual indicators showing the detected dice and their values
    on the input frame. It provides a simple visual representation of the recognition
    results for user feedback.

    The visualization process follows these steps:
    1. Dot Visualization: Draw circles around each detected dot with color coding
    2. Dice Boundary Visualization: Draw rectangles around each detected die
    3. Value Display: Show the numeric value of each die (count of dots)

    Args:
        frame (np.ndarray): Input BGR image frame to draw on
        dice (List["Dice"]): List of detected dice objects

    Returns:
        None: The frame is modified in-place
    """

    # ===== STEP 1: DOT VISUALIZATION =====
    # Draw circles around each detected dot with appropriate color
    for die in dice:
        for blob in die.blobs:
            blob_position = blob.position
            if blob_position is not None:
                # Calculate circle radius based on blob size
                blob_radius = blob.size / 2  # Radius is half the blob size

                # Determine color based on blob type
                #      Blue for dark dots, Red for light dots
                blob_color = DARK_BLOB_COLOR if blob.is_dark else LIGHT_BLOB_COLOR

                # Draw circle at blob position
                cv2.circle(
                    frame,
                    (int(blob_position[0]), int(blob_position[1])),
                    int(blob_radius),
                    blob_color,
                    2,  # Line thickness
                )

    # ===== STEP 2: DICE BOUNDARY VISUALIZATION =====
    # Draw rectangles around each detected die
    for die in dice:
        # Define dice visualization parameters
        # Fixed size rectangle representing the die
        DIE_DISPLAY_SIZE = 80  # Pixels
        die_center = die.position
        half_size = DIE_DISPLAY_SIZE // 2

        # Calculate rectangle coordinates
        top_left_corner = (die_center[0] - half_size,
                           die_center[1] - half_size)
        bottom_right_corner = (
            die_center[0] + half_size, die_center[1] + half_size)

        # Draw yellow rectangle around the die
        cv2.rectangle(
            frame,
            (int(top_left_corner[0]), int(top_left_corner[1])),
            (int(bottom_right_corner[0]), int(bottom_right_corner[1])),
            (0, 255, 255),  # Yellow color
            2,  # Line thickness
        )

        # ===== STEP 3: VALUE DISPLAY =====
        # Calculate text size for centering the value
        die_value_text = str(len(die.blobs))
        text_dimensions = cv2.getTextSize(
            die_value_text,
            cv2.FONT_HERSHEY_PLAIN,  # Font face
            3,  # Font scale
            2,  # Line thickness
        )[0]

        # Draw centered green text showing die value
        text_position_x = int(
            die_center[0] - text_dimensions[0] / 2
        )  # Center horizontally
        text_position_y = int(
            die_center[1] + text_dimensions[1] / 2
        )  # Center vertically

        cv2.putText(
            frame,
            die_value_text,
            (text_position_x, text_position_y),
            cv2.FONT_HERSHEY_PLAIN,
            3,  # Font scale
            (0, 255, 0),  # Green color
            2,  # Line thickness
        )


def __main__():
    """Main entry point for the dice recognition application.

    This function initializes the camera and blob detectors, then runs the main
    processing loop that captures frames, processes them through the dice recognition
    pipeline, and displays the results in real-time.

    The application follows these steps:
    1. Initialization: Set up camera and blob detectors
    2. Frame Processing: Continuously process frames through the recognition pipeline
    3. Cleanup: Release resources when the application is terminated

    The application runs until the user presses 'q' to quit.
    """
    # ===== STEP 1: INITIALIZATION =====
    # Initialize video capture from the webcam
    if IS_WINDOWS:
        video_capture = cv2.VideoCapture(WINDOWS_CAMERA_INDEX)
    else:
        video_capture = cv2.VideoCapture(LINUX_CAMERA_PATH)

    # ===== STEP 2: FRAME PROCESSING LOOP =====
    # Process frames continuously until terminated
    while True:
        # Capture frame from camera
        frame_captured, current_frame = video_capture.read()

        # Check if frame was successfully captured
        if not frame_captured:
            print("Failed to grab frame from camera")
            break

        # Detect dots (blobs) in the frame
        detected_blobs = get_blobs_from_frame(current_frame)

        # Group dots into dice and determine their values
        detected_dice = get_dice_from_blobs(detected_blobs, current_frame)

        # Add visual overlays to highlight detection results
        overlay_info(current_frame, detected_dice)

        # Display the processed frame with recognition results
        cv2.imshow("Dice Recognition", current_frame)

        # Check for user input (1ms timeout)
        key_pressed = cv2.waitKey(1)
        if key_pressed & 0xFF == ord("q"):  # Press 'q' to quit
            break

    # ===== STEP 3: CLEANUP =====
    # Release camera resources
    video_capture.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()


# Start the program
__main__()
