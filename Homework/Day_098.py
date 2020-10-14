#!/usr/bin/env python
# coding: utf-8

# ## Generator 可以使用 next 來進行循環中的一步
# 文字上有點難解釋，直接來看範例就能了解什麼是 Generator!

# ### 撰寫一個 Generator，一次吐出 list 中的一個值

# In[10]:


def output_from_list_generator(your_list):
    for i in your_list:
        yield i 


# In[11]:


my_list = [1, 2, 3, 4, 5]


# In[12]:


gen = output_from_list_generator(my_list)


# In[13]:


print(next(gen))


# In[14]:


print(next(gen))


# In[15]:


print(next(gen))


# In[16]:


print(next(gen))


# In[17]:


print(next(gen))


# In[18]:


# print(next(gen))


# ### 從上面的範例程式碼我們可以看到，當使用一次 next，generator 就會跑 for_loop 一次，因此得到 list 中的第一個值，當再使用一次後，for_loop 記得上次的循環，所以吐出第二個值。最後一次，因為 for loop 已經執行結束了，所以再使用 next 就會看到 StopIteration，無法在得到值

# ### 我們可以撰寫一個無限循環的 Generator，只要使用 While True 即可

# In[20]:


def inf_loop_generator(your_list):
    while True:
        for i in your_list:
            yield i


# In[21]:


gen = inf_loop_generator(my_list)


# In[22]:


print(next(gen))


# In[23]:


print(next(gen))


# In[24]:


print(next(gen))


# In[25]:


print(next(gen))


# In[26]:


print(next(gen))


# In[27]:


print(next(gen))


# In[28]:


print(next(gen))


# ### 上面的程式碼因為我們使用了 While True，所以 for loop 不會結束，只要 call next 就一定會跑一次循環，並返回值

# In[ ]:





# ## 雖然 Cifar-10 的資料可以全部讀進記憶體，但讓我們試著用 Generator，批次的把 Cifar 10 的資料取出來，一次取 32 張出來！

# In[50]:


def img_combine(img, ncols=8, size=1, path=False):
    from math import ceil
    import matplotlib.pyplot as plt
    import numpy as np
    nimg = len(img)
    nrows = int(ceil(nimg/ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True, figsize=(ncols*size,nrows*size))
    if nrows == 0:
        return
    elif ncols == 1:
        for r, ax in zip(np.arange(nrows), axes):
            nth=r
            if nth < nimg:
                ax.imshow(img[nth], cmap='rainbow', vmin=0, vmax=1)
                
            ax.set_axis_off()
    elif nrows == 1:
        for c, ax in zip(np.arange(ncols), axes):
            nth=c
            if nth < nimg:
                ax.imshow(img[nth], cmap='rainbow', vmin=0, vmax=1)
            ax.set_axis_off()
    else:
        for r, row in zip(np.arange(nrows), axes):
            for c, ax in zip(np.arange(ncols), row):
                nth=r*ncols+c
                if nth < nimg:
                    ax.imshow(img[nth], cmap='rainbow', vmin=0, vmax=1)
                ax.set_axis_off()
    plt.show()


# In[44]:


from keras.datasets import cifar10


# In[45]:


(x_train, x_test), (y_train, y_test) = cifar10.load_data()


# In[46]:


def cifar_generator(image_array, batch_size=32):
    while True:
        for indexs in range(0, len(image_array), batch_size):
            images = x_train[indexs: indexs+batch_size]
            labels = x_test[indexs: indexs+batch_size]
            yield images, labels


# In[47]:


cifar_gen = cifar_generator(x_train)


# In[48]:


images, labels = next(cifar_gen)


# In[51]:


img_combine(images)


# In[52]:


images, labels = next(cifar_gen)


# In[54]:


img_combine(images)


# ## 可以看到兩次的圖片並不一樣，這樣就可以開始訓練囉！

# ## 作業

# 請參考昨天的程式碼，將訓練資料讀取方式改寫成 Generator，並將原本的 model.fit 改為 model.fit_generator 來進行訓練。請參考 Keras [官方文件中 fit_generator 的說明](https://keras.io/models/sequential/)
