import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from src.app.dag import SelfHealingDAG
from src.app.config import Config
from pathlib import Path

app = Flask(__name__)
CORS(app)

model_path = str(Config.CHECKPOINTS_DIR / "model")
dag = None

def init_model():
    global dag
    if dag is None:
        print("Loading Self-Healing Classification System...")
        dag = SelfHealingDAG(
            model_path=model_path,
            interactive=False,
            device="cpu"
        )
        print("Model loaded successfully!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if dag is None:
            init_model()
        
        result = dag.run(text)
        
        response = {
            'input_text': text,
            'predicted_label': result['final_label'],
            'confidence': round(result['confidence'] * 100, 2),
            'status': result['status'],
            'probabilities': {k: round(v * 100, 2) for k, v in result['probs'].items()},
            'fallback_activated': result.get('fallback_activated', False),
            'fallback_strategy': result.get('fallback_strategy'),
            'decision_via': result.get('decision_via', 'direct_prediction'),
            'request_id': result['request_id']
        }
        
        if result.get('backup_model'):
            response['backup_model'] = {
                'label': result['backup_model']['label'],
                'confidence': round(result['backup_model']['confidence'] * 100, 2)
            }
        
        return jsonify(response)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model_loaded': dag is not None})

if __name__ == '__main__':
    print("Starting Self-Healing Classification Web App...")
    print(f"Server will be available at http://0.0.0.0:5000")
    print("Model will load on first classification request...")
    app.run(host='0.0.0.0', port=5000, debug=False)
