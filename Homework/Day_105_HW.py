#!/usr/bin/env python
# coding: utf-8

# # 教學目標:
#     
# 回顧 CNN 網路

# # 範例說明:
#     
# 使用 keras 預載的模型
# 
# 使用 keras VGG16 預訓練的權重
# 
# 了解預測後的結果輸出

# # 作業:
# 
#     回答 Q&A

# In[ ]:


from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

#載入預訓練模型
model = VGG16(weights='imagenet', include_top=False)

 # VGG 現存模型要找到一張名為elephant.jpg做處理的預設路徑
img_path = 'elephant.jpg'
#載入影像
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

#執行預測
features = model.predict(x)
print(features)
# decode_predictions 輸出5個最高概率：(類別, 語義概念, 預測概率


# # 問題:
# 
# 為什麼在CNNs中激活函數選用ReLU，而不用sigmoid或tanh函數？

# In[ ]:

# (a)引入非線性函數作為激勵函數，不再是輸入的線性組合，可以逼近任意函數 
# (b)採用ReLU激活函數，整個過程的計算量節省很多; 
# (c)ReLU會使一部分神經元的輸出為0，這樣就造成了網路的稀疏性，並且減少了參數的相互依存關系，緩解了過擬合問題的發生


