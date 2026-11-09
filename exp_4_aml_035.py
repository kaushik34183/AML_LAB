import numpy as np
import matplotlib.pyplot as plt

def predict(x,w,b):
  return w*x+b

def compute_cost(y, y_pred):
  return np.sum((y-y_pred)**2)/n # cost function for actual gradient descent

def gradient(x,y,y_pred):
  dw = (np.sum(x*(y-y_pred)*-2))/n
  db = (np.sum(y-y_pred)*-2)/n
  return dw, db

def parameter_update(w,b,dw,db,alpha):
  w = w - alpha*dw
  b = b - alpha*db
  return w,b

x = np.array([1,2,3], dtype=float)
y = np.array([2,3,4], dtype=float)

n = len(x)
print("Input x : ", x)
print("Actual y : ", y)
print("Number of samples : ", n)

w = 0.0
b = 0.0
alpha = 0.1
itterations = 3
print("Initial weight (w) : ", w)
print("Initial bias (b) : ", b)
print("Learning rate : ", alpha)

y_pred = predict(x,w,b)
print("Predictions : ", y_pred)


cost = compute_cost(y,y_pred)
print("\nInitial prediction : ", y_pred)
print(f"Initial Cost : {cost:.4f}")

costs = []
w_history = []
b_history = []


for i in range(itterations):
  y_pred_current_params = predict(x,w,b)
  dw, db = gradient(x,y,y_pred_current_params)
  w, b = parameter_update(w,b,dw,db,alpha)
  y_pred_after_update = predict(x,w,b)
  cost_after_update = compute_cost(y,y_pred_after_update)

  costs.append(cost_after_update)
  w_history.append(w)
  b_history.append(b)
  print("\n"+"-"*30)
  print(f"Iteration {i+1}")
  print(f"The y_predict (after update) is {np.round(y_pred_after_update, 4)}")
  print(f"The cost (after update) is {cost_after_update:.4f}")
  print(f"Gradients dw: {dw:.4f}, db: {db:.4f}")
  print(f"The updated weight (w) is {w:.4f}")
  print(f"The updated bias (b) is {b:.4f}")


final_w = w_history[-1]
final_b = b_history[-1]
print("\nThe final equation is : ")
print(f"y = {final_w:.4f}x + {final_b:.4f}")
print("The final cost is ", costs[-1])
plt.plot(range(itterations), costs)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function VS Iterations')
plt.show()


plt.scatter(x,y, label='Actual Data')
if itterations > 0:
    final_y_pred_for_plot = predict(x, final_w, final_b)
else:
    final_y_pred_for_plot = predict(x, w, b)
plt.plot(x, final_y_pred_for_plot, color='red', label='Regression Line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression Fit')
plt.legend()
plt.show()