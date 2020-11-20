import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Loading the saved Model
model = keras.models.load_model('model.h5')

# Preparing the Test Generator
test_datagen = ImageDataGenerator(rescale=1. / 255)
# Flow Testing images in batches of 1 using test_datagen
test_generator = test_datagen.flow_from_directory("/Dataset/test",
                                                  batch_size=1,
                                                  class_mode='categorical',
                                                  target_size=(150, 150),
                                                  shuffle=False)

# To Evaluate the model with test images
score = model.evaluate(test_generator, batch_size=1, verbose=1)

# T0 print the Classification Report
y_pred = model.predict(test_generator)
y_pred = np.argmax(y_pred, axis=1)
print(classification_report(test_generator.classes, y_pred))

# To print the Confusion Matrix
cf_matrix = confusion_matrix(test_generator.classes, y_pred)
Labels = ['Brown spot', 'Leaf smut', 'Bacterial leaf blight']
plt.figure(figsize=(8, 8))
heatmap = sns.heatmap(cf_matrix, xticklabels=Labels, yticklabels=Labels, annot=True, fmt='d', color='blue')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Confusion Matrix')
plt.show()