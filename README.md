# Face-recognition-and-attendance-project

---

### Project Overview
1. Purpose:
   - Automate attendance tracking in classrooms, offices, or events.
   - Save time and reduce manual effort.

2. Key Features:
   - Detects faces using the webcam in real-time.
   - Recognizes known faces by comparing them to pre-encoded face data.
   - Logs attendance in a CSV file, including the timestamp of each entry.
   - Displays live video with bounding boxes and names around recognized faces.

---

### Core Components
1. Input:
   - Images Folder: A folder (ImageAttendace) containing photos of individuals with filenames as their names (e.g., John.jpg).
   - Webcam Feed: Captures real-time video for face detection and recognition.

2. Processing:
   - Face Detection: Uses the face_recognition library to locate faces in the images and webcam feed.
   - Face Encoding: Creates numerical representations (embeddings) of faces for comparison.
   - Face Matching: Compares real-time detected faces with known encodings to identify matches.

3. Output:
   - Attendance Log: Appends recognized names and timestamps to an attendance.csv file.
   - Visual Feedback: Displays the webcam feed with bounding boxes and names for recognized faces.

---

### Technologies Used
1. Libraries:
   - cv2 (OpenCV): For webcam access and image processing.
   - face_recognition: For face detection, encoding, and recognition.
   - numpy: For handling numerical data efficiently.

2. File Formats:
   - Image Files: Input images of known individuals (e.g., JPG, PNG).
   - CSV File: Logs attendance data in a structured format.

---

### How It Works
1. Pre-Processing:
   - The program loads all images from the ImageAttendace folder.
   - Extracts names from image filenames.
   - Encodes faces in these images for recognition.

2. Real-Time Recognition:
   - Captures frames from the webcam.
   - Detects faces in the current frame and encodes them.
   - Compares these encodings to the pre-encoded data of known faces.
   - If a match is found, marks the person's attendance in attendance.csv.

3. Output:
   - Displays live webcam feed with visual indicators (bounding boxes and names).
   - Updates the attendance.csv file for new entries.

---

### Applications
- Educational Institutions: Automates student attendance during classes.
- Corporate Offices: Tracks employee attendance.
- Event Management: Verifies participant presence.

---
