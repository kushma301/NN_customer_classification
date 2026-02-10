# Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model

<img width="1209" height="799" alt="546327715-2a210679-0a23-4590-9a97-84b5380f5907" src="https://github.com/user-attachments/assets/dbea2bb1-9ae8-4343-88cf-5ad9b268cfee" />


## DESIGN STEPS

### STEP 1: Data Collection and Understanding
Collect customer data from the existing market and identify the features that influence customer segmentation. Define the target variable as the customer segment (A, B, C, or D).

### STEP 2: Data Preprocessing
Remove irrelevant attributes, handle missing values, and encode categorical variables into numerical form. Split the dataset into training and testing sets.

### STEP 3: Model Design and Training
Design a neural network classification model with suitable input, hidden, and output layers. Train the model using the training data to learn patterns for customer segmentation.

### STEP 4: Model Evaluation and Prediction
Evaluate the trained model using test data and use it to predict the customer segment for new customers in the target market.


### Name: KUSHMA s
### Register Number: 212224040328

```
  # Define Neural Network(Model1)
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 4)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
      

```
```
# Initialize the Model, Loss Function, and Optimizer
model = PeopleClassifier(input_size=X_train.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)
train_model(model, train_loader, criterion, optimizer, epochs=100)

```
```
#function to train the model
def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
          optimizer.zero_grad()
          outputs=model(inputs)
          loss=criterion(outputs, labels)
          loss.backward()
          optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
````


## Dataset Information
<img width="1365" height="339" alt="546775119-8ee14cd4-880c-416e-af23-e353d9a1297e" src="https://github.com/user-attachments/assets/b8192677-a7c2-4f50-a816-3d77f3a42211" />



## OUTPUT




### Confusion Matrix

<img width="811" height="590" alt="546778481-6f77944f-5329-4013-a77d-1bcfa2a5ed4f" src="https://github.com/user-attachments/assets/87d0bffe-f61c-41e9-8864-f5571e16e13b" />

### Classification Report
<img width="1468" height="769" alt="image" src="https://github.com/user-attachments/assets/be4d426c-2c94-4a8e-b28a-b593dd0114d2" />



### New Sample Data Prediction

<img width="1233" height="345" alt="546778922-67e20d6e-f347-4b60-99f9-6bab2fdfe957" src="https://github.com/user-attachments/assets/0840dcbf-89e9-4036-ba3d-2f7960dd535c" />



## RESULT
Thus neural network classification model is developded for the given dataset.
