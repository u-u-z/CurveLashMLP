import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_curve_data(A_points, B_points, n_samples=50):
    """
    准备曲线训练数据
    
    Args:
        A_points (np.array): A曲线的采样点
        B_points (np.array): B曲线的采样点
        n_samples (int): 采样点数量
    
    Returns:
        X: A曲线位置参数
        y: B曲线的切向和法向偏移量
    
    Formula ref: README.md#2.1-Curve-Data-Preparation
    """
    # 计算A曲线的位置参数（弧长归一化）
    A_diffs = np.diff(A_points, axis=0)
    A_lengths = np.sqrt(np.sum(A_diffs**2, axis=1))
    A_cumsum = np.cumsum(A_lengths)
    A_cumsum = np.insert(A_cumsum, 0, 0)
    
    # 防止除以零
    total_length = A_cumsum[-1]
    if total_length < 1e-10:  # 如果总长度太小
        print("警告：曲线总长度接近零")
        total_length = 1e-10
        
    A_params = A_cumsum / total_length
    
    # 计算B曲线相对于A曲线的偏移量
    offsets = []
    for i in range(len(A_points)):
        # 计算A曲线在该点的切向量
        if i == len(A_points) - 1:
            tangent = A_points[i] - A_points[i-1]
        else:
            tangent = A_points[i+1] - A_points[i]
            
        # 防止零向量
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-10:
            print(f"警告：第{i}个点的切向量接近零")
            tangent = np.array([1e-10, 0])
            tangent_norm = np.linalg.norm(tangent)
            
        tangent = tangent / tangent_norm
        
        # 计算法向量（垂直于切向量）
        normal = np.array([-tangent[1], tangent[0]])
        
        # 计算B点相对于A点的偏移
        diff = B_points[i] - A_points[i]
        
        # 分解为切向和法向分量
        tangential_offset = np.dot(diff, tangent)
        normal_offset = np.dot(diff, normal)
        
        offsets.append([tangential_offset, normal_offset])
    
    offsets = np.array(offsets)
    
    # 检查是否有无效值
    if np.any(np.isnan(A_params)) or np.any(np.isnan(offsets)):
        print("警告：数据中包含NaN值")
        print("A_params统计:", np.nanmin(A_params), np.nanmax(A_params))
        print("offsets统计:", np.nanmin(offsets), np.nanmax(offsets))
    
    return A_params.reshape(-1, 1), offsets

def create_model():
    """
    创建神经网络模型
    
    Formula ref: README.md#3.2-Neural-Network-Model
    """
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(1,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(2)  # 输出切向和法向偏移量
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    return model

def train_curve_model(A_curves, B_curves, epochs=100, batch_size=32):
    """
    训练曲线预测模型
    
    Args:
        A_curves (list): A曲线点列表
        B_curves (list): B曲线点列表
        epochs (int): 训练轮数
        batch_size (int): 批次大小
    
    Returns:
        model: 训练好的模型
        history: 训练历史
    
    Formula ref: README.md#3.3-Curve-Model-Training
    """
    # 准备数据
    X_all = []
    y_all = []
    
    for A_points, B_points in zip(A_curves, B_curves):
        X, y = prepare_curve_data(A_points, B_points)
        X_all.append(X)
        y_all.append(y)
    
    X_all = np.vstack(X_all)
    y_all = np.vstack(y_all)
    
    # 数据标准化
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    X_scaled = X_scaler.fit_transform(X_all)
    y_scaled = y_scaler.fit_transform(y_all)
    
    # 分割训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )
    
    # 创建和训练模型
    model = create_model()
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=1
    )
    
    return model, history, X_scaler, y_scaler

def predict_B_curve(model, A_points, X_scaler, y_scaler):
    """
    预测B曲线
    
    Args:
        model: 训练好的模型
        A_points (np.array): A曲线点
        X_scaler: 输入数据的标准化器
        y_scaler: 输出数据的标准化器
    
    Returns:
        np.array: 预测的B曲线点
    
    Formula ref: README.md#3.4-Curve-Prediction
    """
    # 准备A曲线参数
    A_diffs = np.diff(A_points, axis=0)
    A_lengths = np.sqrt(np.sum(A_diffs**2, axis=1))
    A_cumsum = np.cumsum(A_lengths)
    A_cumsum = np.insert(A_cumsum, 0, 0)
    
    # 防止除以零
    total_length = A_cumsum[-1]
    if total_length < 1e-10:
        print("警告：预测时曲线总长度接近零")
        total_length = 1e-10
        
    A_params = A_cumsum / total_length
    
    # 检查并打印调试信息
    print("A_params范围:", np.min(A_params), np.max(A_params))
    
    # 标准化输入
    X_scaled = X_scaler.transform(A_params.reshape(-1, 1))
    print("X_scaled范围:", np.min(X_scaled), np.max(X_scaled))
    
    # 预测偏移量
    offsets_scaled = model.predict(X_scaled)
    print("预测值范围:", np.min(offsets_scaled), np.max(offsets_scaled))
    
    offsets = y_scaler.inverse_transform(offsets_scaled)
    print("反标准化后范围:", np.min(offsets), np.max(offsets))
    
    # 重建B曲线点
    B_points = []
    for i in range(len(A_points)):
        if i == len(A_points) - 1:
            tangent = A_points[i] - A_points[i-1]
        else:
            tangent = A_points[i+1] - A_points[i]
            
        # 防止零向量
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-10:
            print(f"警告：预测时第{i}个点的切向量接近零")
            tangent = np.array([1e-10, 0])
            tangent_norm = np.linalg.norm(tangent)
            
        tangent = tangent / tangent_norm
        normal = np.array([-tangent[1], tangent[0]])
        
        # 使用预测的偏移量重建B点
        B_point = (A_points[i] + 
                  offsets[i][0] * tangent + 
                  offsets[i][1] * normal)
        B_points.append(B_point)
    
    B_points = np.array(B_points)
    
    # 检查最终结果
    if np.any(np.isnan(B_points)):
        print("警告：预测结果包含NaN值")
        print("无效值位置:", np.where(np.isnan(B_points)))
    
    return B_points
