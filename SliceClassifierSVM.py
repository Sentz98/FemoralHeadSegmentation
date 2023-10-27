import os
import json
import numpy as np
from skimage import io, transform
from sklearn import svm
from sklearn.preprocessing import StandardScaler

class SliceClassifierSVM:
    def __init__(self, model_path=None):
        self.model = None
        self.scaler = StandardScaler()
        if model_path:
            self.load_model(model_path)

    def load_images(self, json_file):
        
        if not os.path.exists(json_file):
            print(f"Labels file {json_file} does not exist. Please create it first.")
            return
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        for entry in data:
            subj_folder = os.path.join(entry["folder"], entry["subj"])
            for modality in ["axial", "coronal", "sagittal"]:
                for image_file in os.listdir(os.path.join(subj_folder, modality)):
                    img_path = os.path.join(subj_folder, image_file)
                    image = io.imread(img_path)
                    
                    # TODO check if image contains dx or sx in the name (or none for agittal images) and 
                    # access the correct interval and save the image in the correct part of the dataframe

                    # Check if the image coordinates fall within the good image intervals
                    is_good_image = False
                    intervals = entry.get(modality, [])
                    for interval in intervals:
                        if interval[0] <= image.shape[0] <= interval[1] and \
                            interval[0] <= image.shape[1] <= interval[1]:
                            is_good_image = True
                            break
                    
                    self.image_paths.append(img_path)
                    self.labels.append(is_good_image)

    def train(self, data_dir):
        X, y = self.load_images(data_dir, 'label.json')

        X_flatten = X.reshape(X.shape[0], -1)
        X_scaled = self.scaler.fit_transform(X_flatten)

        self.model = svm.SVC(kernel='linear', C=1)
        self.model.fit(X_scaled, y)

    def save_model(self, model_path):
        if self.model:
            import joblib
            joblib.dump(self.model, model_path)
            print(f"Model saved to {model_path}")
        else:
            print("No model to save. Train a model first.")

    def load_model(self, model_path):
        import joblib
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            print(f"Model loaded from {model_path}")
        else:
            print("Model file does not exist. Train a model or provide a valid model path.")

    def classify(self, image_path):
        if self.model:
            image = io.imread(image_path)
            image = transform.resize(image, (100, 100))
            image_flatten = image.reshape(1, -1)
            image_scaled = self.scaler.transform(image_flatten)
            prediction = self.model.predict(image_scaled)
            if prediction == 5:  # "everything else" label
                return "everything else"
            else:
                return self.class_names[prediction[0]]
        else:
            print("No model loaded. Train a model or load an existing one.")

# Example usage:
# classifier = ImageClassifierSVM(class_names=['axialDX', 'axialSX', 'coronalSX', 'coronalDX', 'saggital'])
# classifier.train('path_to_training_data_folder')
# classifier.save_model('svm_model.pkl')
# classifier.load_model('svm_model.pkl')
# result = classifier.classify('path_to_new_image.jpg')
# print(f"Image classified as: {result}")
