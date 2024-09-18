import cv2
import numpy as np

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    # Apply adaptive thresholding
    thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    return thresholded

def recognize_sudoku_board(preprocessed_image):
    # Find contours in the image
    contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Find the largest contour which should be the Sudoku board
    largest_contour = max(contours, key=cv2.contourArea)
    # Approximate the contour to a polygon
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    # If the polygon has 4 vertices, it is the Sudoku board
    if len(approx) == 4:
        return approx
    else:
        return None

def extract_digits(recognized_board, preprocessed_image):
    # Warp the perspective to get a top-down view of the Sudoku board
    pts1 = np.float32([recognized_board[0], recognized_board[1], recognized_board[2], recognized_board[3]])
    pts2 = np.float32([[0, 0], [450, 0], [450, 450], [0, 450]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(preprocessed_image, matrix, (450, 450))
    # Split the warped image into a 9x9 grid
    grid = np.array_split(warped, 9, axis=0)
    digits = []
    for row in grid:
        row_digits = np.array_split(row, 9, axis=1)
        for digit in row_digits:
            digits.append(digit)
    return digits

def main(image_path):
    preprocessed_image = preprocess_image(image_path)
    recognized_board = recognize_sudoku_board(preprocessed_image)
    if recognized_board is not None:
        digits = extract_digits(recognized_board, preprocessed_image)
        print("Sudoku board recognized and digits extracted.")
    else:
        print("Failed to recognize Sudoku board.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python sudoku_recognition.py <path_to_image>")
    else:
        main(sys.argv[1])
