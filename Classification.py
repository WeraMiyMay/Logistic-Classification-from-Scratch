import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset (features)
data_x = np.array([(7.2, 2.5), (6.4, 2.2), (6.3, 1.5), (7.7, 2.2), (6.2, 1.8), 
                   (5.7, 1.3), (7.1, 2.1), (5.8, 2.4), (5.2, 1.4), (5.9, 1.5), 
                   (7.0, 1.4), (6.8, 2.1), (7.2, 1.6), (6.7, 2.4), (6.0, 1.5), 
                   (5.1, 1.1), (6.6, 1.3), (6.1, 1.4), (6.7, 2.1), (6.4, 1.8), 
                   (5.6, 1.3), (6.9, 2.3), (6.4, 1.9), (6.9, 2.3), (6.5, 2.2), 
                   (6.0, 1.5), (5.6, 1.1), (5.6, 1.5), (6.0, 1.0), (6.0, 1.8), 
                   (6.7, 2.5), (7.7, 2.3), (5.5, 1.1), (5.8, 1.0), (6.9, 2.1), 
                   (6.6, 1.4), (6.3, 1.6), (6.1, 1.4), (5.0, 1.0), (7.7, 2.0), 
                   (4.9, 1.7), (7.2, 1.8), (6.8, 1.4), (6.1, 1.2), (5.8, 1.9), 
                   (6.3, 2.5), (5.7, 2.0), (6.5, 1.8), (7.6, 2.1), (6.3, 1.5), 
                   (6.7, 1.4), (6.4, 2.3), (6.2, 2.3), (6.3, 1.9), (5.5, 1.3), 
                   (7.9, 2.0), (6.7, 1.8), (6.4, 1.3), (6.5, 2.0), (6.5, 1.5), 
                   (6.9, 1.5), (5.6, 1.3), (5.8, 1.2), (6.7, 2.3), (6.0, 1.6), 
                   (5.7, 1.2), (5.7, 1.0), (5.5, 1.0), (6.1, 1.4), (6.3, 1.8), 
                   (5.7, 1.3), (6.1, 1.3), (5.5, 1.3), (6.3, 1.3), (5.9, 1.8), 
                   (7.7, 2.3), (6.5, 2.0), (5.6, 2.0), (6.7, 1.7), (5.7, 1.3), 
                   (5.5, 1.2), (5.0, 1.0), (5.8, 1.9), (6.2, 1.3), (6.2, 1.5), 
                   (6.3, 2.4), (6.4, 1.5), (7.4, 1.9), (6.8, 2.3), (5.6, 1.3), 
                   (5.8, 1.2), (7.3, 1.8), (6.7, 1.5), (6.3, 1.8), (6.0, 1.6), 
                   (6.4, 2.1), (6.1, 1.8), (5.9, 1.8), (5.4, 1.5), (4.9, 1.0)])

# Class labels (+1 and -1)
data_y = np.array([1, 1, 1, 1, 1, -1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 
                   1, 1, 1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, 
                   -1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1, 1, 
                   1, -1, -1, 1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 
                   1, -1, -1, -1, -1, -1, 1, -1, -1, -1, -1, 1, 1, 1, 
                   1, -1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, -1, 
                   1, -1, 1, -1, 1, 1, -1, -1, -1])

# Add bias term (column of ones)
data_x = np.hstack((np.ones((data_x.shape[0], 1)), data_x))
# Initialize weights
weights = np.zeros(data_x.shape[1]) 

learning_rate = 0.01 
num_iterations = 5000 

# Log loss function
def log_loss(margin): 
    return np.log2(1 + np.exp(-margin)) 

# Gradient of log loss
def log_loss_gradient(margin): 
    exp_term = np.exp(-margin) 
    return -exp_term / ((1 + exp_term) * np.log(2)) 

# Training loop (gradient descent)
for iteration in range(num_iterations): 
    margins = data_y * np.dot(data_x, weights) 
    loss_gradients = log_loss_gradient(margins) * data_y 
    weight_gradient = np.dot(data_x.T, loss_gradients) / data_x.shape[0] 
    weights -= learning_rate * weight_gradient 

print("Веса модели:", weights) 

# Compute margins for each sample
for i in range(len(data_x)): 
    margin = np.dot(data_x[i], weights) * data_y[i] 
    print(f"Точка {data_x[i][1:]} (метка класса {data_y[i]}): отступ = {margin}") 

# Visualization
data = np.column_stack((data_x[:, 1:], data_y)) 

plt.figure(figsize=(10, 6)) 
sns.kdeplot(x=data[:, 0], y=data[:, 1], hue=data[:, 2], fill=True, alpha=0.5) 
plt.scatter(data_x[data_y == 1][:, 1], data_x[data_y == 1][:, 2], color='blue', label='Класс +1') 
plt.scatter(data_x[data_y == -1][:, 1], data_x[data_y == -1][:, 2], color='red', label='Класс -1') 

# Plot decision boundary
x1 = np.linspace(data_x[:, 1].min() - 1, data_x[:, 1].max() + 1, 100) 
x2 = -(weights[0] + weights[1] * x1) / weights[2] 
plt.plot(x1, x2, color='red', label='Разделяющая линия') 

plt.xlabel('Признак 1') 
plt.ylabel('Признак 2') 
plt.legend() 
plt.grid(True) 
plt.title('Визуализация классификации с разделяющей линией')
plt.show()
