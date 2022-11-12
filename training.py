#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import mylib.model_implementation as model_implementation
import mylib.make_dataloader as make_dataloader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


# In[4]:


num_epochs = 20
train_loss_list, train_acc_list, val_loss_list, val_acc_list = [], [], [], []


# In[5]:


device = "cuda" if torch.cuda.is_available() else "cpu"


# In[6]:


model = model_implementation.cifar10_alexnet().to(device)


# In[7]:


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


# In[8]:


train_loader = make_dataloader.make_train_loader()
test_loader = make_dataloader.make_test_loader()


# In[ ]:


### training
for epoch in range(num_epochs):
    train_loss, train_acc, val_loss, val_acc = 0.0, 0.0, 0.0, 0.0
    
    # ====== train_mode ======
    model.train()
    for i, (images, labels) in enumerate(tqdm(train_loader)):
      images, labels = images.to(device), labels.to(device)      
      optimizer.zero_grad()
      outputs = model(images)
      loss = criterion(outputs, labels)
      train_loss += loss.item()
      train_acc += (outputs.max(1)[1] == labels).sum().item()
      loss.backward()
      optimizer.step()
    
    avg_train_loss = train_loss / len(train_loader.dataset)
    avg_train_acc = train_acc / len(train_loader.dataset)
    
    # ====== val_mode ======
    model.eval()
    with torch.no_grad():
      for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
        val_acc += (outputs.max(1)[1] == labels).sum().item()
    avg_val_loss = val_loss / len(test_loader.dataset)
    avg_val_acc = val_acc / len(test_loader.dataset)
    
    print ('Epoch [{}/{}], Loss: {loss:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}' 
                   .format(epoch+1, num_epochs, i+1, loss=avg_train_loss, val_loss=avg_val_loss, val_acc=avg_val_acc))
    train_loss_list.append(avg_train_loss)
    train_acc_list.append(avg_train_acc)
    val_loss_list.append(avg_val_loss)
    val_acc_list.append(avg_val_acc)

