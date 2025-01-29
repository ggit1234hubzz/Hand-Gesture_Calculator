
# Hand Gesture Calculator

An interactive calculator that uses **MediaPipe** and **OpenCV** to perform mathematical operations (addition, subtraction, multiplication, division) through hand gestures. This project recognizes hand gestures to count the number of fingers displayed, calculates the result based on the inputs from both hands, and shows the total value in real-time.

## Features:
- **Hand Gesture Recognition:** Detects and counts the fingers on both hands.
- **Real-Time Calculation:** Adds the values from both hands to calculate a total.
- **Interactive Interface:** Displays real-time finger count and total value on the screen.

## Technologies Used:
- **Python**: Main programming language.
- **MediaPipe**: Hand gesture detection library.
- **OpenCV**: Computer vision library for processing video input and rendering results.

## Installation:

1. Clone this repository:
   ```bash
   git clone https://github.com/fxjrin/hand-gesture-calculator.git
   ```
2. Install required dependencies:
   ```bash
   pip install mediapipe opencv-python
   ```

3. Run the program:
   ```bash
   python calculator.py
   ```

## How It Works:
- The system uses the webcam to detect hand gestures and count the number of fingers displayed on each hand.
- It computes the total finger count from both hands and displays the result on the screen in real-time.

## Contributing:
Feel free to fork this repository, create an issue, or submit a pull request with improvements!

## License:
This project is open-source and available under the [MIT License](LICENSE).
