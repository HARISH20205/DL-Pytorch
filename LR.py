import torch
import torch.nn as nn
import numpy as np

class LinearRegression:
    def __init__(self):
        self.model = None
        
    def fit(self, X,y,epochs = 1000, learning_rate = 0.001):
        X = torch.tensor(X,dtype = torch.float32)
        y = torch.tensor(y,dtype=torch.float32).view(-1,1)
        
        self.model = nn.Linear(X.shape[1],1)
        
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(self.model.parameters(),lr = learning_rate)
        
        for epoch in range(epochs):
            y_pred = self.model(X)
            loss = loss_fn(y_pred,y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
    
    def predict(self,X):
        X = torch.tensor(X,dtype=torch.float32)
        
        with torch.no_grad():
            y_pred = self.model(X)
        return y_pred.numpy()
    
if __name__ == "__main__":
    # X = np.array([[1], [2], [3], [4]], dtype=np.float32)  
    # y = np.array([2.2, 4.0, 5.9, 8.1], dtype=np.float32)  
    np.random.seed(0)
    X = np.random.rand(100, 1) * 10  
    y = 3.5 * X.squeeze() + np.random.randn(100) * 0.5 

    regressor = LinearRegression()
    regressor.fit(X, y, epochs=10000, learning_rate=0.01)

    # Make predictions
    X_test = np.array([[5], [6]], dtype=np.float32)
    predictions = regressor.predict(X_test)
    print("Predictions:", predictions)  