import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model

img_width, img_height = 300, 300

model = load_model('trained_weights.h5')  # 选取自己的.h模型名称
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

img = cv2.imread('./test/pos/7.jpg')
# img = cv2.imread('anquanmao2.jpg')
img = cv2.resize(img, (img_width, img_height))
img = img.astype("float") / 255.0
img_ = img.copy()
img = np.reshape(img, [1, img_width, img_height, 3])
pred = model.predict(img)

predictedClass = "UNRECOGNIZABLE"
if pred <= 0.5:
    predictedClass = "No hardhat"
else:
    predictedClass = "Hardhat"

cv2.putText(img_, predictedClass, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
plt.imshow(img_)
plt.show()

print("The predicted class is: ", predictedClass)
print("The model's predicted score is: ", pred[0][0])
