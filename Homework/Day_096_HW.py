#!/usr/bin/env python
# coding: utf-8

# In[3]:


from keras.layers import Conv2D, SeparableConv2D, Input
from keras.models import Model


# In[9]:


input_image = Input((224, 224, 3))
feature_maps = Conv2D(filters=32, kernel_size=(3,3))(input_image)
feature_maps2 = Conv2D(filters=64, kernel_size=(3,3))(feature_maps)
model = Model(inputs=input_image, outputs=feature_maps2)


# In[11]:


model.summary()


# ## 可以看到經過兩次 Conv2D，如果沒有設定 padding="SAME"，圖就會越來越小，同時特徵圖的 channel 數與 filters 的數量一致

# In[12]:


input_image = Input((224, 224, 3))
feature_maps = SeparableConv2D(filters=32, kernel_size=(3,3))(input_image)
feature_maps2 = SeparableConv2D(filters=64, kernel_size=(3,3))(feature_maps)
model = Model(inputs=input_image, outputs=feature_maps2)


# In[13]:


model.summary()


# ## 可以看到使用 Seperable Conv2D，即使模型設置都一模一樣，但是參數量明顯減少非常多！

# ## 作業

# 請閱讀 Keras 官方範例 [mnist_cnn.py](https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py)  
# 
# 並回答下列問題。僅有 70 行程式碼，請確保每一行的程式碼你都能夠理解目的
# 
# 1. 是否有對資料做標準化 (normalization)? 如果有，在哪幾行?  37.38
# 2. 使用的優化器 Optimizer 為何? Adadelta
# 3. 模型總共疊了幾層卷積層? 3
# 4. 模型的參數量是多少? so many
# 

# In[ ]:





