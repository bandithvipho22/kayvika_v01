import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
import time
import zmq

class SignLanguageRecognizer:
    SEQUENCE_LENGTH = 30
    # ACTIONS = ['hello', 'I', 'go', 'home', 'like', 'today', 'want', 'meet', 'you', 'drink']
    ACTIONS = ['hello', 'I', 'go', 'home', 'like', 'today', 'want', 'meet', 'you', 'drink', 'how are you?', 'toilet', 'all of you', 'again', 'sad']

    def __init__(self, model_path='results_v05/sign_model.tflite'):
        self.interpreter = self._load_interpreter(model_path)
        self.hands, self.mp_hands, self.mp_drawing = self._initialize_mediapipe()
        self.sequence_data = deque(maxlen=self.SEQUENCE_LENGTH)
        self.predicted_sign = ""
        self.last_prediction_time = 0
        self.prediction_interval = 0.4  # Predict every 0.4 seconds
        self.sentence = []
        self.last_sign = ""
        self.sign_threshold = 2  # Number of consecutive predictions to confirm a sign
        self.sign_count = 0
        self.ready_for_prediction = False
        
        # Setup ZeroMQ context and socket for communication
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.connect("tcp://localhost:5555")  # Connect to the translation service

    def _load_interpreter(self, model_path):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        self.input_details = interpreter.get_input_details()
        self.output_details = interpreter.get_output_details()
        return interpreter

    def _initialize_mediapipe(self):
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        return hands, mp_hands, mp.solutions.drawing_utils

    def process_frame(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image)
        return image, results

    def extract_landmarks(self, hand_landmarks):
        return np.array([[landmark.x, landmark.y, landmark.z] for landmark in hand_landmarks.landmark]).flatten()

    def predict_sign(self):
        sequence_data = np.array(list(self.sequence_data)).astype(np.float32)
        sequence_data = np.expand_dims(sequence_data, axis=0)
        self.interpreter.set_tensor(self.input_details[0]['index'], sequence_data)
        self.interpreter.invoke()
        prediction = self.interpreter.get_tensor(self.output_details[0]['index'])
        return self.ACTIONS[np.argmax(prediction[0])]
    
    def fix_misrecognition(self, predicted_sign, hand_count, points):
        if predicted_sign in ["you", "go"]:
            if predicted_sign == "go" and hand_count == 1:
                predicted_sign = "you"
            elif predicted_sign == "you" and hand_count == 2:
                predicted_sign = "go"
        return predicted_sign

    def prepare_sentence(self, predicted_sign):
        if predicted_sign != self.last_sign:
            self.last_sign = predicted_sign
            self.sign_count = 1
        else:
            self.sign_count += 1

        if self.sign_count == self.sign_threshold:
            if predicted_sign not in self.sentence:
                self.sentence.append(predicted_sign)

    def complete_and_send_sentence(self):
        if self.sentence:
            completed_sentence = ' '.join(self.sentence)
            print("Completed Sentence:", completed_sentence)
            self.socket.send_string(completed_sentence)  # Send sentence to the translation service
            self.sentence = []  # Clear the sentence

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            image, results = self.process_frame(frame)
            
            landmarks_frame = np.zeros(63)

            hand_count = 0
            if results.multi_hand_landmarks:
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    landmarks = self.extract_landmarks(hand_landmarks)
                    if i == 0:
                        self.sequence_data.append(landmarks)
                        if len(self.sequence_data) == self.SEQUENCE_LENGTH:
                            self.ready_for_prediction = True
                    hand_count += 1

            else:
                # Clear sequence data if no hands are detected
                self.sequence_data.clear()
                self.ready_for_prediction = False
  
            current_time = time.time()
            if self.ready_for_prediction and (current_time - self.last_prediction_time >= self.prediction_interval):
                self.predicted_sign = self.predict_sign()
                self.predicted_sign = self.fix_misrecognition(self.predicted_sign, hand_count, landmarks)
                self.prepare_sentence(self.predicted_sign)
                self.last_prediction_time = current_time

            # Update the displayed sentence
            displayed_sentence = ' '.join(self.sentence)

            cv2.putText(frame, f"Sign: {self.predicted_sign}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Sentence: {displayed_sentence}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow('Sign Language Recognition', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space key triggers sending the sentence
                self.complete_and_send_sentence()

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    recognizer = SignLanguageRecognizer()
    recognizer.run()
