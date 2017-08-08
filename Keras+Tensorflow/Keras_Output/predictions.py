from keras.models import load_model
import numpy as np 
import os,sys
from PIL import Image
import matplotlib.pyplot as plot


model = load_model('mnist_keras_DNN_model.h5')

# take input to test it 
URL = "test/test2.JPG"
img = Image.open(URL)
print img.bits, img.size, img.format

# Resize the image
img = img.resize((28,28),Image.ANTIALIAS)
#img.save("test/image_scaled.jpg",quality=95)

# Gray Scale Image 

def grayscale(picture):
    res= Image.new(picture.mode, picture.size)
    width, height = picture.size

    for i in range(0, width):
        for j in range(0, height):
            pixel=picture.getpixel((i,j))
            avg=(pixel[0]+pixel[1]+pixel[2])/3

            # Do Supression 
            if avg > 120 : 
            	avg = 255 
            avg = 255 - avg
            res.putpixel((i,j),(avg,avg,avg))
    res.show()
    return res

gray_image = grayscale(img)


# Print Image
def printImage(picture):
	width, height = picture.size

	for j in range(0, height):
		for i in range(0, width):
			pixel = picture.getpixel((i,j))
			print pixel[0], 
		print('\n')


printImage(gray_image)


# Normalize between 0 and 1 
def normalize(picture):
	width, height = picture.size
	normalized_array = []

	for j in range(0, height):
		for i in range(0, width):
			pixel = picture.getpixel((i,j))
			normalized_array.append( pixel[0] / 255.0 )
	return np.array(normalized_array)


X_test = normalize(gray_image)
print(X_test)
X_test = X_test.reshape(1, 784)  # X_test.reshape(1, 28, 28, 1) for CNN

# Do predictions 

classes = model.predict(X_test)

print(classes)

classValue = classes[0].max()
print(classValue)
indexVal = np.where(classes[0]==classValue)
print(indexVal)







