from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime
import json
import requests

# MySQL 参考: 权限管理系统 使用 PyMySQL 进行连接
import pymysql
from pymysql import Error

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 加载保存的模型
MODEL_PATH = None
model = None

# 加载环境变量中的 BigModel API Key（可选）
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

BIGMODEL_API_KEY = os.getenv("BIGMODEL_API_KEY")
BIGMODEL_CHAT_URL = os.getenv(
    "BIGMODEL_CHAT_URL",
    "https://open.bigmodel.cn/api/paas/v4/chat/completions"
)
BIGMODEL_MODEL = os.getenv("BIGMODEL_MODEL", "glm-4-flash")

# 数据库配置（参考 权限管理系统）
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '123456',
    'database': 'my_db',
    'charset': 'utf8mb4'
}

def get_db_connection():
    """获取数据库连接，使用 DictCursor 便于字段访问"""
    try:
        conn = pymysql.connect(
            host=DB_CONFIG['host'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            database=DB_CONFIG['database'],
            charset=DB_CONFIG['charset'],
            cursorclass=pymysql.cursors.DictCursor
        )
        return conn
    except Error as e:
        print(f"✗ 数据库连接失败: {e}")
        return None

def init_db():
    """初始化心脏病预测日志表(若不存在则创建)"""
    conn = get_db_connection()
    if not conn:
        return False
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS heart_predictions (
                    id INT PRIMARY KEY AUTO_INCREMENT,
                    age INT NULL,
                    sex TINYINT NULL,
                    chest_pain_type VARCHAR(16) NULL,
                    resting_bp INT NULL,
                    cholesterol INT NULL,
                    fasting_bs TINYINT NULL,
                    resting_ecg VARCHAR(16) NULL,
                    max_hr INT NULL,
                    exercise_angina TINYINT NULL,
                    oldpeak DOUBLE NULL,
                    st_slope VARCHAR(16) NULL,
                    prediction TINYINT NULL,
                    healthy_prob DOUBLE NULL,
                    heart_disease_prob DOUBLE NULL,
                    confidence DOUBLE NULL,
                    risk_level VARCHAR(16) NULL,
                    model_file VARCHAR(255) NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    raw_input TEXT
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
                """
            )
        conn.commit()
        return True
    except Error as e:
        print(f"✗ 初始化日志表失败: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def to_int_safe(x):
    try:
        return int(x)
    except Exception:
        return None

def to_float_safe(x):
    try:
        return float(x)
    except Exception:
        return None

def map_yes_no(v):
    # FastingBS / ExerciseAngina: 是->1, 否->0
    if v in (1, 0):
        return int(v)
    if isinstance(v, str):
        return 1 if v.strip() in ['是', 'Y', 'y', 'Yes', 'YES'] else 0
    return 0

def map_sex(v):
    # Sex: M->1, F->0
    if v in (1, 0):
        return int(v)
    if isinstance(v, str):
        v2 = v.strip().upper()
        return 1 if v2 == 'M' else 0
    return 0

def log_heart_prediction(raw_input: dict, result: dict):
    """将一次预测的原始输入与模型输出写入数据库。失败仅记录日志，不影响接口返回。"""
    conn = get_db_connection()
    if not conn:
        return
    try:
        # 提取并做最小转换（保留原始语义）
        age = to_int_safe(raw_input.get('Age'))
        sex = map_sex(raw_input.get('Sex'))
        chest_pain_type = str(raw_input.get('ChestPainType', ''))[:16]
        resting_bp = to_int_safe(raw_input.get('RestingBP'))
        cholesterol = to_int_safe(raw_input.get('Cholesterol'))
        fasting_bs = map_yes_no(raw_input.get('FastingBS'))
        resting_ecg = str(raw_input.get('RestingECG', ''))[:16]
        max_hr = to_int_safe(raw_input.get('MaxHR'))
        exercise_angina = map_yes_no(raw_input.get('ExerciseAngina'))
        oldpeak = to_float_safe(raw_input.get('Oldpeak'))
        st_slope = str(raw_input.get('ST_Slope', ''))[:16]

        prediction = int(result.get('prediction')) if result and 'prediction' in result else None
        healthy_prob = float(result['probability']['healthy']) if result and 'probability' in result else None
        heart_prob = float(result['probability']['heart_disease']) if result and 'probability' in result else None
        confidence = float(result.get('confidence')) if result and 'confidence' in result else None
        risk_level = str(result.get('risk_level', ''))[:16]
        model_file = str(result.get('model_info', {}).get('model_file', ''))[:255]
        raw_json = json.dumps(raw_input, ensure_ascii=False)

        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO heart_predictions (
                    age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs,
                    resting_ecg, max_hr, exercise_angina, oldpeak, st_slope,
                    prediction, healthy_prob, heart_disease_prob, confidence,
                    risk_level, model_file, raw_input
                ) VALUES (
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s
                )
                """,
                (
                    age, sex, chest_pain_type, resting_bp, cholesterol, fasting_bs,
                    resting_ecg, max_hr, exercise_angina, oldpeak, st_slope,
                    prediction, healthy_prob, heart_prob, confidence,
                    risk_level, model_file, raw_json
                )
            )
        conn.commit()
        print("🗄️ 已写入一次心脏病预测记录到数据库")
    except Error as e:
        print(f"✗ 写入预测日志失败: {e}")
        conn.rollback()
    except Exception as ex:
        print(f"✗ 写入预测日志发生异常: {ex}")
        conn.rollback()
    finally:
        conn.close()

def load_latest_heart_model():
    """加载最新的心脏病预测模型"""
    global MODEL_PATH, model
    
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 优先查找balanced_models目录下的SMOTE模型
    balanced_models_dir = os.path.join(script_dir, 'balanced_models')
    
    # 查找指定的心脏病预测模型文件
    target_model = 'smote_catboost_balanced_20250903_145140.pkl'
    model_path = os.path.join(balanced_models_dir, target_model)
    
    if os.path.exists(model_path):
        MODEL_PATH = model_path
        try:
            # 使用joblib加载模型
            model = joblib.load(MODEL_PATH)
            print(f"🎉 成功加载心脏病预测模型: {MODEL_PATH}")
            return True, f"心脏病预测模型加载成功: {os.path.basename(MODEL_PATH)}"
        except Exception as e:
            return False, f"模型加载失败: {str(e)}"
    else:
        return False, f"未找到指定的心脏病预测模型文件: {target_model}"

def prepare_heart_input_data(data):
    """准备心脏病预测输入数据，确保特征顺序和格式正确"""
    
    # 创建DataFrame
    input_df = pd.DataFrame([data])
    
    # 数值型特征处理（确保数据类型正确）
    numeric_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
    for feature in numeric_features:
        if feature in input_df.columns:
            input_df[feature] = pd.to_numeric(input_df[feature], errors='coerce').fillna(0).astype('float64')
        else:
            input_df[feature] = 0.0
    
    # 二元分类特征编码（与训练时保持一致）
    # Sex: M=1, F=0
    if 'Sex' in input_df.columns:
        input_df['Sex'] = input_df['Sex'].map({'M': 1, 'F': 0}).fillna(0).astype('int64')
    else:
        input_df['Sex'] = 0
    
    # FastingBS: 是=1, 否=0
    if 'FastingBS' in input_df.columns:
        input_df['FastingBS'] = input_df['FastingBS'].map({'是': 1, '否': 0}).fillna(0).astype('int64')
    else:
        input_df['FastingBS'] = 0
    
    # ExerciseAngina: 是=1, 否=0
    if 'ExerciseAngina' in input_df.columns:
        input_df['ExerciseAngina'] = input_df['ExerciseAngina'].map({'是': 1, '否': 0}).fillna(0).astype('int64')
    else:
        input_df['ExerciseAngina'] = 0
    
    # 多分类特征独热编码（按训练时的顺序）
    # ChestPainType 独热编码
    chest_pain_type = input_df.get('ChestPainType', ['ASY'])[0] if 'ChestPainType' in input_df.columns else 'ASY'
    
    # 按字母顺序创建独热编码（通常pandas的get_dummies是按字母顺序的）
    input_df['ChestPainType_ASY'] = 1 if chest_pain_type == 'ASY' else 0
    input_df['ChestPainType_ATA'] = 1 if chest_pain_type == 'ATA' else 0
    input_df['ChestPainType_NAP'] = 1 if chest_pain_type == 'NAP' else 0
    input_df['ChestPainType_TA'] = 1 if chest_pain_type == 'TA' else 0
    
    # RestingECG 独热编码
    resting_ecg = input_df.get('RestingECG', ['Normal'])[0] if 'RestingECG' in input_df.columns else 'Normal'
    
    input_df['RestingECG_LVH'] = 1 if resting_ecg == 'LVH' else 0
    input_df['RestingECG_Normal'] = 1 if resting_ecg == 'Normal' else 0
    input_df['RestingECG_ST'] = 1 if resting_ecg == 'ST' else 0
    
    # ST_Slope 独热编码
    st_slope = input_df.get('ST_Slope', ['Up'])[0] if 'ST_Slope' in input_df.columns else 'Up'
    
    input_df['ST_Slope_Down'] = 1 if st_slope == 'Down' else 0
    input_df['ST_Slope_Flat'] = 1 if st_slope == 'Flat' else 0
    input_df['ST_Slope_Up'] = 1 if st_slope == 'Up' else 0
    
    # 删除原始的分类特征列
    columns_to_drop = ['ChestPainType', 'RestingECG', 'ST_Slope']
    for col in columns_to_drop:
        if col in input_df.columns:
            input_df = input_df.drop(col, axis=1)
    
    # 按照训练时的确切特征顺序排列
    # 这个顺序是通过模拟训练时的数据处理过程得出的
    expected_features = [
        'Age', 'Sex', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'ExerciseAngina', 'Oldpeak',
        'ChestPainType_ASY', 'ChestPainType_ATA', 'ChestPainType_NAP', 'ChestPainType_TA',
        'RestingECG_LVH', 'RestingECG_Normal', 'RestingECG_ST',
        'ST_Slope_Down', 'ST_Slope_Flat', 'ST_Slope_Up'
    ]
    
    # 确保所有特征都存在
    for feature in expected_features:
        if feature not in input_df.columns:
            input_df[feature] = 0
    
    # 按照期望的顺序排列特征
    input_df = input_df[expected_features]
    
    # 确保数据类型正确
    for col in input_df.columns:
        if col in ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']:
            input_df[col] = input_df[col].astype('float64')
        else:
            input_df[col] = input_df[col].astype('int64')
    
    return input_df

@app.route('/')
def home():
    """返回HTML页面"""
    try:
        # 获取当前脚本所在目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        html_path = os.path.join(script_dir, 'app_heart.html')
        
        with open(html_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        # 获取当前脚本所在目录用于调试
        script_dir = os.path.dirname(os.path.abspath(__file__))
        return f"""
        <h1>心脏病预测系统</h1>
        <p>❌ app_heart.html 文件未找到</p>
        <p>📂 当前脚本目录: {script_dir}</p>
        <p>🔍 寻找文件: {os.path.join(script_dir, 'app_heart.html')}</p>
        <p>📋 请确保 app_heart.html 文件在同一目录下</p>
        """

@app.route('/predict', methods=['POST'])
def predict():
    """心脏病预测接口"""
    try:
        # 确保日志表可用（初始化一次，失败不影响预测）
        init_db()
        # 检查模型是否已加载
        if model is None:
            success, message = load_latest_heart_model()
            if not success:
                return jsonify({
                    'success': False,
                    'error': message,
                    'suggestion': '请检查模型文件是否存在'
                }), 500
        
        # 获取请求数据
        data = request.json
        print(f"📥 收到预测请求: {data}")
        
        # 数据验证
        required_fields = ['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 
                          'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 
                          'Oldpeak', 'ST_Slope']
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'success': False,
                'error': f'缺少必要字段: {missing_fields}'
            }), 400
        
        # 准备输入数据
        input_data = prepare_heart_input_data(data)
        print(f"🔧 处理后的输入数据形状: {input_data.shape}")
        print(f"🔧 处理后的输入数据:\n{input_data}")
        print(f"🔧 输入特征名称: {list(input_data.columns)}")
        
        # 尝试获取模型期望的特征名称
        try:
            if hasattr(model, 'feature_names_'):
                print(f"🎯 模型期望的特征名称: {model.feature_names_}")
            elif hasattr(model, 'get_feature_names'):
                print(f"🎯 模型期望的特征名称: {model.get_feature_names()}")
            else:
                print("⚠️ 无法获取模型的特征名称")
        except Exception as e:
            print(f"⚠️ 获取模型特征名称时出错: {e}")
        
        # 进行预测
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        
        # 解释预测结果
        result = {
            'success': True,
            'prediction': int(prediction),
            'probability': {
                'healthy': float(prediction_proba[0]),
                'heart_disease': float(prediction_proba[1])
            },
            'confidence': float(max(prediction_proba)),
            'risk_level': get_risk_level(prediction_proba[1]),
            'interpretation': get_heart_disease_interpretation(prediction, prediction_proba),
            'model_info': {
                'model_file': os.path.basename(MODEL_PATH) if MODEL_PATH else '未知',
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        print(f"🎯 预测结果: {result}")
        # 写入数据库日志（不影响接口返回）
        try:
            log_heart_prediction(data, result)
        except Exception as _:
            # 任何异常均吞掉，保证接口可用
            pass
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ 预测过程出错: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'预测过程出错: {str(e)}',
            'suggestion': '请检查输入数据格式是否正确'
        }), 500

def get_risk_level(heart_disease_prob):
    """根据心脏病概率返回风险等级"""
    if heart_disease_prob < 0.3:
        return '低风险'
    elif heart_disease_prob < 0.6:
        return '中等风险'
    elif heart_disease_prob < 0.8:
        return '高风险'
    else:
        return '极高风险'

def get_heart_disease_interpretation(prediction, proba):
    """获取心脏病预测结果的解释"""
    heart_disease_prob = proba[1]
    healthy_prob = proba[0]
    
    if prediction == 0:  # 预测为健康
        if healthy_prob > 0.8:
            return f"✅ 心脏健康状况良好 (置信度: {healthy_prob:.1%})"
        else:
            return f"⚠️ 心脏健康，但建议定期检查 (置信度: {healthy_prob:.1%})"
    else:  # 预测为心脏病
        if heart_disease_prob > 0.8:
            return f"🚨 存在心脏病风险，强烈建议立即就医 (置信度: {heart_disease_prob:.1%})"
        else:
            return f"⚠️ 可能存在心脏病风险，建议咨询医生 (置信度: {heart_disease_prob:.1%})"

@app.route('/model_info', methods=['GET'])
def model_info():
    """获取模型信息"""
    if model is None:
        success, message = load_latest_heart_model()
        if not success:
            return jsonify({
                'success': False,
                'error': message
            }), 500
    
    return jsonify({
        'success': True,
        'model_path': MODEL_PATH,
        'model_name': os.path.basename(MODEL_PATH) if MODEL_PATH else '未知',
        'load_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_type': '心脏病预测 - SMOTE平衡CatBoost模型'
    })


@app.route('/ai_chat', methods=['POST'])
def ai_chat():
    """
    AI 助手对话接口：转发到 BigModel 对话补全 API
    请求体: { message: str, history?: [{role, content}] }
    响应: { success: bool, reply?: str, error?: str }
    """
    try:
        if not BIGMODEL_API_KEY:
            return jsonify({
                'success': False,
                'error': '未配置 BIGMODEL_API_KEY，请在环境变量中设置后重试'
            }), 500

        payload = request.get_json(silent=True) or {}
        user_message = (payload.get('message') or '').strip()
        history = payload.get('history') or []

        if not user_message:
            return jsonify({'success': False, 'error': 'message 不能为空'}), 400

        # 构造对话消息，加入系统提示
        messages = [
            {
                'role': 'system',
                'content': '你是一个严谨的健康科普助手，面向心血管健康场景，提供通俗、审慎的建议；明确声明不替代医生诊断。'
            }
        ]

        # 限制历史长度，避免过长
        for m in history[-10:]:
            if isinstance(m, dict) and m.get('role') in ('user', 'assistant') and isinstance(m.get('content'), str):
                messages.append({'role': m['role'], 'content': m['content']})

        messages.append({'role': 'user', 'content': user_message})

        req_body = {
            'model': BIGMODEL_MODEL,
            'messages': messages,
            # 可按需扩展：temperature, top_p 等
        }

        headers = {
            'Authorization': f'Bearer {BIGMODEL_API_KEY}',
            'Content-Type': 'application/json'
        }

        resp = requests.post(
            BIGMODEL_CHAT_URL,
            headers=headers,
            json=req_body,
            timeout=20
        )

        if resp.status_code != 200:
            return jsonify({
                'success': False,
                'error': f'上游接口错误: HTTP {resp.status_code}, {resp.text[:200]}'
            }), 502

        data = resp.json()
        # 兼容 OpenAI 风格返回
        reply = None
        try:
            reply = data['choices'][0]['message']['content']
        except Exception:
            # 兜底解析
            reply = data.get('data') or data.get('output') or data.get('result')

        if not reply:
            reply = '抱歉，暂未获取到有效回复，请稍后再试。'

        return jsonify({'success': True, 'reply': str(reply)})

    except Exception as e:
        return jsonify({'success': False, 'error': f'AI 助手调用失败: {str(e)}'}), 500

if __name__ == '__main__':
    print("🚀 启动心脏病预测系统...")
    print("=" * 50)
    
    # 尝试加载模型
    success, message = load_latest_heart_model()
    if success:
        print(f"✅ {message}")
    else:
        print(f"❌ {message}")
        print("⚠️ 系统将在接收到第一个预测请求时尝试重新加载模型")
    
    print("=" * 50)
    print("🌐 服务器启动信息:")
    print("   - 访问地址: http://localhost:5001")
    print("   - 预测接口: http://localhost:5001/predict")
    print("   - 模型信息: http://localhost:5001/model_info")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5001)
