import coremltools

DNN_ml_model = coremltools.converters.keras.convert('mnist_keras_DNN_model.h5')
DNN_ml_model.author = 'Ashis Laha'
DNN_ml_model.description = 'Use for handwritten digit recognition'
DNN_ml_model.save('mnist_keras_DNN.mlmodel')
print(DNN_ml_model)

CNN_ml_model = coremltools.converters.keras.convert('mnist_keras_CNN_model.h5')
CNN_ml_model.author = 'Ashis Laha'
CNN_ml_model.description = 'Use for handwritten digit recognition'
CNN_ml_model.save('mnist_keras_CNN.mlmodel')
print(CNN_ml_model)