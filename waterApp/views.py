
from django.shortcuts import render
import numpy as np
# Create your views here.

from django.core.files.storage import FileSystemStorage

from keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator, load_img,img_to_array 
import json
import tensorflow as tf
from tensorflow import Graph

img_height, img_width=150,150
d={"0":"algae","1":"clean","2":"mud","3":"polluted"}

model_graph=Graph()
with model_graph.as_default():
    gpuoptions = tf.compat.v1.GPUOptions(allow_growth=True)
    tf_session= tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpuoptions))
    with tf_session.as_default():
        model =load_model('./models/best_model.h5')

def index(request):
    context={'a':1}
    return render(request,'index.html',context)



def predictImage(request):
    print(request)
    print(request.POST.dict())
    fileObj=request.FILES['filePath']
    fs=FileSystemStorage()
    filePathName=fs.save(fileObj.name,fileObj)
    filePathName=fs.url(filePathName)
    print(filePathName)
    testimage='.'+ filePathName
    img=load_img(testimage,target_size=(img_height,img_width))
    x=img_to_array(img)
    x=x/255
    x=x.reshape(1,img_height, img_width,3)
    input_arr=np.array([x])
    input_arr.shape
    with model_graph.as_default():
       with tf_session.as_default():
            predi=model.predict(x)

    predictedLabel = d[str(np.argmax(predi))]

    context={'filePathName':filePathName,'predictedLabel':predictedLabel}
    return render(request,'index.html',context)

def viewDataBase(request):
    import os
    listOfImages=os.listdir('./media/')
    listOfImagesPath=['./media/'+i for i in listOfImages]
    context={'listOfImagesPath':listOfImagesPath}
    return render(request,'viewDB.html',context)