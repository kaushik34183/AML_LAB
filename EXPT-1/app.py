from flask import Flask, request, jsonify, send_from_directory
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__, static_folder='.', static_url_path='')

def calculate_linear_regression(x, y):
    n = 4
    sumX, sumY = sum(x), sum(y)
    sumXY = sum(x[i] * y[i] for i in range(n))
    sumXX = sum(x[i]**2 for i in range(n))
    
    denom = (n * sumXX) - (sumX**2)
    if denom == 0:
        raise ValueError("Denominator is 0.")
        
    m = ((n * sumXY) - (sumX * sumY)) / denom
    c = (sumY - m * sumX) / n
    return m, c

def calculate_metrics(y, y_pred):
    n = 4
    mae = sum(abs(y[i] - y_pred[i]) for i in range(n)) / n
    mse = sum((y[i] - y_pred[i])**2 for i in range(n)) / n
    rmse = mse**0.5
    
    mean_y = sum(y) / n
    ss_res = sum((y[i] - y_pred[i])**2 for i in range(n))
    ss_tot = sum((y[i] - mean_y)**2 for i in range(n))
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1.0
    
    return mae, mse, rmse, r_squared

def generate_metrics_graph(x, y, m, c, mae, mse, rmse, r_squared):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    
    # Left subplot: Regression Fit Line
    ax1.scatter(x, y, color='blue', label='Data Points')
    x_line = [min(x) - 0.5, max(x) + 0.5]
    y_line = [m * val + c for val in x_line]
    ax1.plot(x_line, y_line, color='red', label=f'y = {m:.4f}x + {c:.4f}')
    ax1.set_title('Linear Regression')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.legend()
    ax1.grid(True)
    
    # Right subplot: Model Metrics
    metrics_labels = ['MSE', 'MAE', 'RMSE', 'R2']
    metrics_values = [mse, mae, rmse, r_squared]
    ax2.scatter(metrics_labels, metrics_values, color='purple')
    for i, val in enumerate(metrics_values):
        ax2.text(metrics_labels[i], val, f"{val:.4f}", va='bottom', ha='center')
    ax2.set_title('Model Metrics')
    ax2.set_xlabel('Metrics')
    ax2.set_ylabel('Value')
    ax2.grid(True)
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    try:
        data = request.get_json() or {}
        x, y = data.get('x', []), data.get('y', [])
        if len(x) != 4 or len(y) != 4:
            return jsonify({'status': 'error', 'message': 'Exactly 4 data points are required.'}), 400
        
        m, c = calculate_linear_regression(x, y)
        y_pred = [m * val + c for val in x]
        mae, mse, rmse, r_squared = calculate_metrics(y, y_pred)
        graph_base64 = generate_metrics_graph(x, y, m, c, mae, mse, rmse, r_squared)
        
        return jsonify({
            'status': 'success',
            'm': m, 'c': c,
            'mae': mae, 'mse': mse, 'rmse': rmse, 'r_squared': r_squared,
            'y_pred': y_pred,
            'metrics_graph': graph_base64
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8000)