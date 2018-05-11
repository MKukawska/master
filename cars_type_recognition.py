from scipy.io import loadmat
import os
import numpy as np
from PIL import Image
from sklearn.utils import shuffle
from collections import Counter
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.models import Model
from keras.regularizers import l2
import matplotlib.pyplot as plt
import tensorflow as tf

'''
Importing data
'''

def read_anno(matrix, exclude):
    classNum_tech=[]
    classNumber_org=[]
    className=[]
    fileName=[]
    train_test_split=[]
    for idx in range(matrix['annotations'].shape[1]):
        classNum_tech.append(matrix['annotations'][0][idx][5][0][0])
        sel = matrix['class_names'][:,classNum_tech[idx]-1][0][0].split()[-2]
        if sel in (exclude):
            continue
        else:
            className.append(sel)
            classNumber_org.append(matrix['annotations'][0][idx][5][0][0]) # index 5 has class number
            fileName.append(matrix['annotations'][0][idx][0][0].split('/')[1]) # index 0 has the file name
            train_test_split.append(matrix['annotations'][0][idx][6][0][0])
    return classNumber_org, className, fileName, train_test_split

def train_test(fileName, train_test_split, className):
    fileName_x_train = []
    fileName_x_test = []
    className_y_train = []
    className_y_test = [] 
    for i in range(len(fileName)):
        if train_test_split[i] == 0:
            fileName_x_train.append(fileName[i])
            className_y_train.append(className[i])
        else:
            fileName_x_test.append(fileName[i])
            className_y_test.append(className[i])
    return fileName_x_train, fileName_x_test, className_y_train, className_y_test

def dict_classes(className):
    obs_by_class = Counter(className)
    classes = list(set(className))
    classToNumber = {}
    for i in range(len(classes)):
        classToNumber[classes[i]] = i
    return classToNumber, obs_by_class
    
def transform_class(className, classToNumber):
    classNumber = np.zeros([len(className), len(classToNumber)], 'float32')
    for i in range(len(className)):
        classNumber[i, classToNumber[className[i]]] = 1
    return classNumber
    
def prepare_labels(matrix, exclude):    
    classNumber_org, className, fileName, train_test_split = read_anno(matrix, exclude)
    #fileName_x_train, fileName_x_test, className_y_train, className_y_test = train_test(fileName, train_test_split, className)
    classToNumber, obs_by_class = dict_classes(className)
    y_train = transform_class(className, classToNumber)
    #y_test = transform_class(className_y_test, classToNumber)
    return y_train, fileName, obs_by_class, classToNumber

def prepare_images(fileName, img_rows, img_cols):
    x = np.empty([len(fileName), img_rows, img_cols], 'float32')
    for i in range(len(fileName)):
        im = Image.open(work_dir + '\car_ims\\' + fileName[i])   
        img = im.resize((img_rows,img_cols))
        grey = np.array(img.convert('L'), 'float32')
        grey /= 255
        x[i] = grey
    return x

def prepare_datasets(matrix, img_rows, img_cols, exclude):
    y_train, fileName, obs_by_class, classToNumber = prepare_labels(matrix, exclude)
    x_train = prepare_images(fileName, img_rows, img_cols)
    #x_test = prepare_images(fileName_x_test, img_rows, img_cols)
    x_train, y_train = shuffle(x_train, y_train)
    #x_test, y_test = shuffle(x_test, y_test)
    return x_train, y_train, obs_by_class, classToNumber

work_dir=r"C:\Users\Magda\Desktop\CARS"

# I wanted to choose only 3 types that significantly differ from one another and are numerous enough. I chose Sedan, Convertible and SUV
exclude = ['Minivan', 'Hatchback', 'Coupe', 'Abarth', 'Cab', 'GS', 'IPL', 'R', 'SRT-8', 'SRT8', 'SS', 'SuperCab', 'Superleggera', 'Type-S', 'Van', 'Wagon', 'XKR', 'Z06', 'ZR1']
img_rows,img_cols = 64, 64
matrix = loadmat(os.path.join( work_dir, "cars_annos.mat"))

x_train, y_train, obs_by_class, classToNumber = prepare_datasets(matrix, img_rows, img_cols, exclude)


'''
Building the model
'''

sess = tf.Session()

L2 = 0.01

x_train1 = x_train.reshape(x_train.shape + (1,))
#x_test1 = x_test.reshape(x_test.shape + (1,))

with tf.device('/device:GPU:0'):
    model_input = Input(shape=(img_rows, img_cols, 1))
    
    aux = model_input
    
    aux = Conv2D(4, 3, activation='relu', padding='same', kernel_regularizer=l2(L2),)(aux)
    aux = MaxPooling2D()(aux)
    aux = Conv2D(8, 3, activation='relu', padding='valid', kernel_regularizer=l2(L2),)(aux)
    aux = MaxPooling2D()(aux)
    aux = Conv2D(16, 3, activation='relu', padding='valid', kernel_regularizer=l2(L2),)(aux)
    aux = Flatten()(aux)
    aux = Dense(200, activation='relu')(aux)
    aux = Dropout(0.5)(aux)
    output = Dense(3, activation='softmax')(aux)
    
    model = Model(model_input, output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    
    hist = model.fit(x_train1, y_train, batch_size=32, epochs=50, verbose=1, validation_split=0.2, class_weight='auto')

train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(50)

plt.style.use(['classic'])

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('Epochs')
plt.ylabel('loss')
plt.title('Loss Function (crossentropy)')
plt.grid(True)
plt.legend(['train','val'],loc=2)

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('prediction')
plt.grid(True)
plt.legend(['train','val'],loc=2)