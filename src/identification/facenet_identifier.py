import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import cv2
from PIL import Image

class FacenetIdentifier:
    def __init__(self, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.encoder = LabelEncoder()
        self.knn = KNeighborsClassifier(n_neighbors=5)
        self.embeddings = []
        self.labels = []

    def add_face(self, image_path, label):
        image = Image.open(image_path)
        faces = self.mtcnn(image)
        if faces is not None:
            for face in faces:
                face = face.unsqueeze(0).to(self.device)
                embedding = self.resnet(face).detach().cpu().numpy().flatten()
                self.embeddings.append(embedding)
                self.labels.append(label)
            print(f"Added face(s) with label: {label}")
        else:
            print("No face detected in the image.")

    def train(self):
        if len(self.embeddings) > 0:
            embeddings_array = np.array(self.embeddings)
            labels_array = np.array(self.labels)
            labels_encoded = self.encoder.fit_transform(labels_array)
            self.knn.fit(embeddings_array, labels_encoded)
            print("Training completed.")
        else:
            print("No faces to train on.")

    def identify_face(self, image_path):
        image = Image.open(image_path)
        faces = self.mtcnn(image)
        if faces is not None:
            for face in faces:
                face = face.unsqueeze(0).to(self.device)
                embedding = self.resnet(face).detach().cpu().numpy().flatten()
                prediction = self.knn.predict([embedding])
                probabilities = self.knn.predict_proba([embedding])
                label = self.encoder.inverse_transform(prediction)
                label_prob = probabilities[0][prediction[0]]
                print(f"Identified face as: {label[0]} with probability: {label_prob:.2f}")
                return label[0], label_prob
        else:
            print("No face detected in the image.")
            return None, None

# Example usage
if __name__ == "__main__":
    face_identifier = FacenetIdentifier()

    # Add faces to the database
    face_identifier.add_face("path/to/image1.jpg", "Person1")
    face_identifier.add_face("path/to/image2.jpg", "Person2")
    
    # Train the KNN classifier
    face_identifier.train()

    # Identify a new face
    label, probability = face_identifier.identify_face("path/to/new_image.jpg")
    print(f"Label: {label}, Probability: {probability}")

