import cv2

#python3 ~/Desktop/ImageRecog1/ImageRecog.py


def detect_objects():
    # Load pre-trained face, eye, and profile face detection models (Haar cascades)
    face_cascade = cv2.CascadeClassifier('/Users/jung/Desktop/ImageRecog1/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('/Users/jung/Desktop/ImageRecog1/haarcascade_eye.xml')
    profile_cascade = cv2.CascadeClassifier('/Users/jung/Desktop/ImageRecog1/haarcascade_profileface.xml')

    # Set up other necessary configurations
    min_face_size = (30, 30)
    min_eye_size = (10, 10)
    min_profile_size = (30, 30)
    scale_factor = 1.3

    # Start video capture
    cap = cv2.VideoCapture(0)  # 0 for default camera

    while True:
        # Read the current frame from the video feed
        ret, frame = cap.read()

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Perform face detection using the face cascade model
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minSize=min_face_size)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Extract the region of interest (ROI) within the face region
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            # Perform eye detection within the face region
            eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=scale_factor, minSize=min_eye_size)

            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
                cv2.putText(roi_color, "Eye", (ex, ey-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Perform profile face detection within the face region
            profiles = profile_cascade.detectMultiScale(roi_gray, scaleFactor=scale_factor, minSize=min_profile_size)

            for (px, py, pw, ph) in profiles:
                cv2.rectangle(roi_color, (px, py), (px+pw, py+ph), (0, 0, 255), 2)
                cv2.putText(roi_color, "Profile", (px, py-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Display the resulting frame
        cv2.imshow('Object Recognition Robot', frame)

        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

# Call the function to start object detection
detect_objects()
