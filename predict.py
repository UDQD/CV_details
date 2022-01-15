import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt

model = load_model("my_model.h5")
IMG_SHAPE = 500
content = os.listdir('input/')
df = pd.DataFrame(content)
df.rename(columns = {0 : 'filename'},inplace = True)
df['value'] = None
for name in content:
    ind = df[df['filename']==name].index[0]
    img = load_img(f'input/{name}')
    img_array = img_to_array(img)
    img_array = tf.image.resize(img_array, (IMG_SHAPE, IMG_SHAPE))
    img_array = img_array / 255.0
    img_expended = np.expand_dims(img_array, axis=0)
    prediction = round(float(model.predict(img_expended)))
    pred_label = 'small' if prediction == 1 else 'big'
    if pred_label == 'small':
        img.save(f'output/small/{name}')
        df['value'][ind] = 'small'
    else:
        img.save(f'output/big/{name}')
        df['value'][ind] = 'big'
df.to_excel('output_table.xlsx')