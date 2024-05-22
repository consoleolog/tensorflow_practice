import tensorflow as tf
import pandas as pd
import numpy as np
import time

data = pd.read_csv('gpascore.csv')

#빈칸 없애기
# data.isnull().sum()
data = data.dropna()
#data['출력하고싶은열이름]
#data['출력하고싶은열이름].min()
#data['출력하고싶은열이름].count

#전부 리스트로 변경
y데이터 = data['admit'].values

x데이터 = []

for i, rows in data.iterrows():
    #print(rows) 모든 열 출력
    #print(rows['열이름']) 모든 열 출력
    x데이터.append([ rows['gre'], rows['gpa'], rows['rank']]) #[gre,gpa,rank] 하나씩 출력

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, activation= 'tanh'),
    tf.keras.layers.Dense(32, activation= 'tanh'),
    tf.keras.layers.Dense(64, activation= 'tanh'),
    tf.keras.layers.Dense(128, activation = 'tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam' , loss= 'binary_crossentropy', metrics=['accuracy'])




tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./logs/{}'.format("model-"+str(int(time.time()))))

#accuracy면 모드가 Max loss 면 모드는 min
#es = EarlyStopping(monitor='val_loss',patience=10, mode='max')
es = tf.keras.callbacks.EarlyStopping(
    monitor='accuracy',
    patience=10,
    mode='max'
)

model.fit( np.array(x데이터) , np.array(y데이터) , epochs= 1000 ,callbacks=[tensorboard,es])

tf.keras.utils.plot_model(model,to_file='model.png',show_shapes=True,show_layer_names=True)


input = tf.keras.layers.Input()
flatten = tf.keras.layers.Flatten()(input)
dense_1 = tf.keras.layers.Dense()(flatten)
reshape1 = tf.keras.layers.Reshape( (28,28) )(dense_1)

