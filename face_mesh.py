import cv2
import mediapipe as mp

# Initialize MediaPipe Face Mesh and Drawing Utilities
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Set up the Face Mesh model
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,             # Change this to detect more faces
    refine_landmarks=True,       # Adds more detail around eyes and lips
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Start capturing video from the default webcam (0)
cap = cv2.VideoCapture(0)

print("Press 'q' to quit.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a later selfie-view display
    # And convert the color space from BGR (OpenCV default) to RGB (MediaPipe requires RGB)
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance, optionally mark the image as not writeable to pass by reference.
    image.flags.writeable = False
    
    # Process the image and find face landmarks
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw the mesh 
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            # Draw the contours (eyes, lips, face oval)
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )

    # Display the output
    cv2.imshow('Real-Time Face Landmark Detection', image)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()