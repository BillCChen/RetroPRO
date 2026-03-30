from flask import Flask, request, jsonify
import os

app = Flask(__name__, static_folder='.', static_url_path='')

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/api/prediction-control', methods=['POST'])
def prediction_control():
    data = request.get_json()
    action = data.get('action')
    file_path = data.get('filePath')
    parameters = data.get('parameters', {})
    
    try:
        # 确保文件目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # 写入文件
        with open(file_path, 'w') as f:
            f.write(action)
        
        # 如果提供了参数，保存参数到另一个文件
        if parameters:
            param_file = file_path.replace('.txt', '_params.json')
            import json
            with open(param_file, 'w') as f:
                json.dump(parameters, f, indent=2)
        
        return jsonify({
            'success': True,
            'message': f'文件 {file_path} 已更新为 {action}',
            'parameters': parameters
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # 监听所有网络接口，允许外部访问
    app.run(host='0.0.0.0', port=5000, debug=True)