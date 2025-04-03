import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

def load_gaze_data(file_path):
    """CSVファイルから視線データを読み込む"""
    df = pd.read_csv(file_path)
    # #N/Aの値をNaNに置換
    df = df.replace('#N/A', np.nan)
    # NaNの行を削除
    df = df.dropna()
    return df

def create_gaze_plot(df):
    """視線データのプロットを作成"""
    # プロットの設定
    plt.figure(figsize=(12, 8))
    
    # 左目のプロット
    left_points = np.column_stack((df['LX(fix)'], df['LY(fix)']))
    left_segments = np.concatenate([left_points[:-1, None], left_points[1:, None]], axis=1)
    left_collection = LineCollection(left_segments, cmap='Blues', linewidth=1)
    left_collection.set_array(df['TS'])
    plt.gca().add_collection(left_collection)
    
    # 右目のプロット
    right_points = np.column_stack((df['RX(fix)'], df['RY(fix)']))
    right_segments = np.concatenate([right_points[:-1, None], right_points[1:, None]], axis=1)
    right_collection = LineCollection(right_segments, cmap='Reds', linewidth=1)
    right_collection.set_array(df['TS'])
    plt.gca().add_collection(right_collection)
    
    # プロットの装飾
    plt.title('視線軌跡の可視化', fontsize=14, pad=20)
    plt.xlabel('X座標 (ピクセル)', fontsize=12)
    plt.ylabel('Y座標 (ピクセル)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 凡例の追加
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', label='左目'),
        Line2D([0], [0], color='red', label='右目')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # カラーバーの追加
    cbar = plt.colorbar(left_collection, label='時間 (秒)')
    
    # 軸の範囲を自動調整
    plt.autoscale()
    
    # グラフの保存
    plt.savefig('gaze_trajectory.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # データの読み込み
    df = load_gaze_data('gazedata.csv')
    
    # プロットの作成
    create_gaze_plot(df)
    print("視線軌跡の可視化が完了しました。'gaze_trajectory.png'を確認してください。")

if __name__ == "__main__":
    main() 