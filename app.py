from flask import Flask, render_template, request, send_file, session, after_this_request
import pandas as pd
import os
import json
from werkzeug.utils import secure_filename
import tempfile
import numpy as np
import plotly.graph_objects as go
import cv2
from io import BytesIO
from tempfile import NamedTemporaryFile
import uuid
from functools import wraps
from scipy.ndimage import gaussian_filter
from datetime import timedelta

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit
app.secret_key = 'your-secret-key-here'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=1)  # セッションの有効期限を1時間に設定

# 一時データ保存用のディレクトリを作成
TEMP_DATA_DIR = os.path.join(tempfile.gettempdir(), 'gaze_data')
if not os.path.exists(TEMP_DATA_DIR):
    os.makedirs(TEMP_DATA_DIR)

def save_temp_data(data_dict):
    """セッションにデータを保存"""
    session['gaze_data'] = data_dict
    # セッションIDを生成して保存
    session_id = str(uuid.uuid4())
    session['session_id'] = session_id
    return session_id

def load_temp_data(session_id):
    """セッションからデータを読み込み"""
    if 'gaze_data' in session and session.get('session_id') == session_id:
        return session['gaze_data']
    return None

def cleanup_temp_data(session_id):
    """セッションからデータを削除"""
    if 'gaze_data' in session and session.get('session_id') == session_id:
        session.pop('gaze_data')
        session.pop('session_id')

def load_gaze_data(file_path):
    """CSVファイルから視線データを読み込む"""
    # 3行目をヘッダーとして読み込む
    df = pd.read_csv(file_path, header=2)
    df = df.replace('#N/A', np.nan)
    df = df.dropna(subset=['LX', 'LY'])
    
    # データのダウンサンプリング
    if len(df) > 1000:
        sampling_rate = len(df) // 1000
        df = df.iloc[::sampling_rate].copy()
    
    # データを時間でソート
    df = df.sort_values('TS')
    if df['TS'].min() > 0:
        df['TS'] = df['TS'] - df['TS'].min()
    
    return df

def create_gaze_plot(df, title, color='blue', div_id=None, recent_only=False, return_fig=False):
    """視線データのプロットを作成"""
    fig = go.Figure()
    
    # フレームの作成
    frames = []
    time_steps = np.linspace(df['TS'].min(), df['TS'].max(), 50)
    for t in time_steps:
        mask = df['TS'] <= t
        if recent_only:
            # 最新の40秒分のデータのみを表示
            mask = (df['TS'] <= t) & (df['TS'] > max(t - 40, 0))
            # 最新の点のマスク
            latest_mask = df['TS'] <= t
            latest_point = df[latest_mask].iloc[-1:] if len(df[latest_mask]) > 0 else pd.DataFrame()
        
        trace_data = []
        # 軌跡のトレース
        trace_data.append(
            go.Scatter(
                x=df[mask]['LX'],
                y=df[mask]['LY'],
                mode='lines',
                name=title,
                line=dict(color=color, width=1),
                customdata=df[mask]['TS'],
                hovertemplate='X: %{x:.0f}<br>Y: %{y:.0f}<br>時間: %{customdata:.2f}秒<extra></extra>'
            )
        )
        
        # 最新の点のトレース（recent_onlyの場合のみ）
        if recent_only and not latest_point.empty:
            trace_data.append(
                go.Scatter(
                    x=latest_point['LX'],
                    y=latest_point['LY'],
                    mode='markers',
                    name=f'{title} (現在位置)',
                    marker=dict(size=8, color=color),
                    customdata=latest_point['TS'],
                    hovertemplate='X: %{x:.0f}<br>Y: %{y:.0f}<br>時間: %{customdata:.2f}秒<extra></extra>'
                )
            )
        
        frames.append(go.Frame(data=trace_data))
    
    # 初期データの追加
    if recent_only:
        # 最初は空のデータを表示
        fig.add_trace(go.Scatter(
            x=[],
            y=[],
            mode='lines',
            name=title,
            line=dict(color=color, width=1),
            hovertemplate='X: %{x:.0f}<br>Y: %{y:.0f}<br>時間: %{customdata:.2f}秒<extra></extra>'
        ))
        # 最新の点用のトレース
        fig.add_trace(go.Scatter(
            x=[],
            y=[],
            mode='markers',
            name=f'{title} (現在位置)',
            marker=dict(size=8, color=color),
            hovertemplate='X: %{x:.0f}<br>Y: %{y:.0f}<br>時間: %{customdata:.2f}秒<extra></extra>'
        ))
    else:
        fig.add_trace(go.Scatter(
            x=df['LX'],
            y=df['LY'],
            mode='lines',
            name=title,
            line=dict(color=color, width=1),
            customdata=df['TS'],
            hovertemplate='X: %{x:.0f}<br>Y: %{y:.0f}<br>時間: %{customdata:.2f}秒<extra></extra>'
        ))
    
    # レイアウトの設定
    fig.update_layout(
        title=title,
        xaxis_title='X座標 (ピクセル)',
        yaxis_title='Y座標 (ピクセル)',
        showlegend=True,
        width=800,  # さらに幅を拡大
        height=400,  # さらに高さを拡大（1:2の比率を維持）
        xaxis=dict(range=[0, 4320], constrain='domain'),
        yaxis=dict(
            range=[0, 2160],
            scaleanchor="x",
            scaleratio=1.0,
            constrain='domain'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        updatemenus=[{
            'type': 'buttons',
            'showactive': True,
            'x': 1.2,  # タイトルの右側に配置
            'y': 1.1,  # タイトルと同じ高さに配置
            'buttons': [{
                'label': '再生',
                'method': 'animate',
                'args': [None, {'frame': {'duration': 50, 'redraw': True}, 'fromcurrent': True}]
            }, {
                'label': '一時停止',
                'method': 'animate',
                'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}]
            }]
        }]
    )
    
    # 円を追加
    circle_radius = 324 / 2  # 直径から半径を計算
    fig.add_shape(
        type="circle",
        xref="x",
        yref="y",
        x0=2160 - circle_radius,
        y0=1080 - circle_radius,
        x1=2160 + circle_radius,
        y1=1080 + circle_radius,
        line_color="rgba(0,0,0,0.5)",
        line_width=2,
        fillcolor="rgba(0,0,0,0)"
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
    
    # framesをFigureオブジェクトの直接のプロパティとして設定
    fig.frames = frames
    
    # スライダーの設定
    fig.update_layout(
        sliders=[{
            'currentvalue': {'prefix': '時間: ', 'suffix': '秒'},
            'steps': [
                dict(
                    method='restyle',
                    label=f'{t:.1f}秒',
                    args=[{
                        'x': [df[df['TS'] <= t]['LX'].tolist()] if not recent_only else [
                            df[(df['TS'] <= t) & (df['TS'] > max(t - 40, 0))]['LX'].tolist(),
                            [df[df['TS'] <= t].iloc[-1]['LX']] if len(df[df['TS'] <= t]) > 0 else []
                        ],
                        'y': [df[df['TS'] <= t]['LY'].tolist()] if not recent_only else [
                            df[(df['TS'] <= t) & (df['TS'] > max(t - 40, 0))]['LY'].tolist(),
                            [df[df['TS'] <= t].iloc[-1]['LY']] if len(df[df['TS'] <= t]) > 0 else []
                        ]
                    }]
                )
                for t in time_steps
            ]
        }]
    )
    
    if return_fig:
        return fig
    
    if div_id:
        return fig.to_html(full_html=False, div_id=div_id)
    return fig.to_html(full_html=False)

def calculate_distance(df):
    """タイムスタンプごとの視点移動距離を計算"""
    # 座標の差分を計算
    dx = df['LX'].diff()
    dy = df['LY'].diff()
    
    # ユークリッド距離を計算
    distances = np.sqrt(dx**2 + dy**2)
    
    # 統計情報を計算
    stats = {
        '総移動距離': distances.sum(),
        '平均移動距離': distances.mean(),
        '最大移動距離': distances.max(),
        '標準偏差': distances.std(),
        'データ点数': len(df)
    }
    
    return stats

def create_heatmap(df, title, width=400, height=300):
    """視線データのヒートマップを作成"""
    # 2Dヒストグラムを作成（ビンの数を増やして解像度を上げる）
    x_bins = np.linspace(0, 4320, 100)  # ビンの数を50から100に増加
    y_bins = np.linspace(0, 2160, 60)   # ビンの数を30から60に増加
    
    H, xedges, yedges = np.histogram2d(df['LX'], df['LY'], bins=[x_bins, y_bins])
    
    # データをガウシアンフィルタでスムージング
    H_smooth = gaussian_filter(H, sigma=1.5)
    
    # ヒートマップを作成
    fig = go.Figure(data=go.Heatmap(
        z=H_smooth.T,
        x=xedges[:-1],
        y=yedges[:-1],
        colorscale='Viridis',
        colorbar=dict(
            title=dict(
                text='頻度',
                side='right',
                font=dict(size=12)
            )
        ),
        hoverongaps=False,
        hovertemplate='X: %{x:.0f}<br>Y: %{y:.0f}<br>頻度: %{z:.0f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=f'{title}の視線ヒートマップ',
            font=dict(size=16),
            y=0.95
        ),
        xaxis_title='X座標 (ピクセル)',
        yaxis_title='Y座標 (ピクセル)',
        width=800,
        height=600,
        margin=dict(l=60, r=60, t=60, b=60),
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(size=12)
    )
    
    # 軸の設定を調整
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.1)',
                    zeroline=False, showline=True, linewidth=1, linecolor='black')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.1)',
                     zeroline=False, showline=True, linewidth=1, linecolor='black')
    
    return fig.to_html(full_html=False)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file1' not in request.files and 'file2' not in request.files:
            return render_template('index.html', message='ファイルが選択されていません')
        
        dataframes = {}
        filenames = {}
        plot_htmls = []
        plot_htmls_right = []
        heatmaps = []
        distance_stats = {}
        
        # 各ファイルを処理
        for file_key in ['file1', 'file2']:
            file = request.files.get(file_key)
            if file and file.filename:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                try:
                    df = load_gaze_data(filepath)
                    # DataFrameをディクショナリに変換
                    df_dict = {
                        'LX': df['LX'].tolist(),
                        'LY': df['LY'].tolist(),
                        'TS': df['TS'].tolist()
                    }
                    key = f'データ{len(dataframes) + 1}'
                    dataframes[key] = df_dict
                    filenames[key] = filename
                    
                    # 移動距離の統計を計算
                    distance_stats[key] = calculate_distance(df)
                    
                    # 個別のプロットを作成
                    color = 'blue' if file_key == 'file1' else 'red'
                    fig = create_gaze_plot(df, filename, color)
                    plot_htmls.append(fig)
                    
                    # 右側用のプロットを作成
                    fig_right = create_gaze_plot(df, filename, color, f'{file_key}_right', recent_only=True)
                    plot_htmls_right.append(fig_right)
                    
                    # ヒートマップを作成
                    df_pandas = pd.DataFrame({
                        'LX': df_dict['LX'],
                        'LY': df_dict['LY'],
                        'TS': df_dict['TS']
                    })
                    heatmap = create_heatmap(df_pandas, filename)
                    heatmaps.append(heatmap)
                    
                    # 一時ファイルを削除
                    os.remove(filepath)
                except Exception as e:
                    app.logger.error(f'ファイル処理中にエラーが発生: {str(e)}')
                    return render_template('index.html', message=f'エラーが発生しました: {str(e)}')
        
        # データをセッションに保存
        temp_data = {
            'dataframes': dataframes,
            'filenames': filenames,
            'distance_stats': distance_stats
        }
        session_id = save_temp_data(temp_data)
        app.logger.debug(f'セッションID: {session_id}を生成しました')
        
        if len(dataframes) == 2:
            # DataFrameを再構築
            df1 = pd.DataFrame(dataframes['データ1'])
            df2 = pd.DataFrame(dataframes['データ2'])
            reconstructed_dataframes = {
                'データ1': df1,
                'データ2': df2
            }
            
            # 比較プロットの作成
            fig_combined = create_combined_plot(reconstructed_dataframes, filenames, False)
            plot_htmls.insert(0, fig_combined.to_html(full_html=False))
            
            fig_combined_right = create_combined_plot(reconstructed_dataframes, filenames, True)
            
            # 右側のプロットを正しい順序で配置
            plot_htmls_right = [
                fig_combined_right.to_html(full_html=False, div_id='combined_right'),
                create_gaze_plot(df1, filenames['データ1'], 'blue', 'plot1_right', recent_only=True),
                create_gaze_plot(df2, filenames['データ2'], 'red', 'plot2_right', recent_only=True)
            ]
            
            return render_template('index.html',
                                plot1=plot_htmls[1] if len(plot_htmls) > 1 else None,
                                plot2=plot_htmls[2] if len(plot_htmls) > 2 else None,
                                plot3=plot_htmls[0],
                                plot1_right=plot_htmls_right[1],
                                plot2_right=plot_htmls_right[2],
                                plot3_right=plot_htmls_right[0],
                                heatmap1=heatmaps[0] if len(heatmaps) > 0 else None,
                                heatmap2=heatmaps[1] if len(heatmaps) > 1 else None,
                                distance_stats=distance_stats,
                                session_id=session_id)
    
    return render_template('index.html')

def create_combined_plot(dataframes, filenames, recent_only=False):
    """比較プロットを作成する関数"""
    fig = go.Figure()
    
    # フレームの作成
    frames = []
    max_time = max(
        dataframes['データ1']['TS'].max(),
        dataframes['データ2']['TS'].max()
    )
    
    time_steps = np.linspace(0, max_time, 50)
    for t in time_steps:
        trace_data = []
        for i, (key, df) in enumerate(dataframes.items(), 1):
            if recent_only:
                # 最新の40秒分のデータのみを表示
                mask = (df['TS'] <= t) & (df['TS'] > max(t - 40, 0))
                # 最新の点のマスク
                latest_mask = df['TS'] <= t
                latest_point = df[latest_mask].iloc[-1:] if len(df[latest_mask]) > 0 else pd.DataFrame()
            else:
                mask = df['TS'] <= t
            
            color = 'blue' if i == 1 else 'red'
            
            # 軌跡のトレース
            trace_data.append(
                go.Scatter(
                    x=df[mask]['LX'],
                    y=df[mask]['LY'],
                    mode='lines' if recent_only else 'lines',  # 常にlines
                    name=filenames[key],
                    line=dict(color=color, width=1),
                    customdata=df[mask]['TS'],
                    hovertemplate='X: %{x:.0f}<br>Y: %{y:.0f}<br>時間: %{customdata:.2f}秒<extra></extra>'
                )
            )
            
            # 最新の点のトレース（recent_onlyの場合のみ）
            if recent_only and not latest_point.empty:
                trace_data.append(
                    go.Scatter(
                        x=latest_point['LX'],
                        y=latest_point['LY'],
                        mode='markers',
                        name=f'{filenames[key]} (現在位置)',
                        marker=dict(size=8, color=color),
                        customdata=latest_point['TS'],
                        hovertemplate='X: %{x:.0f}<br>Y: %{y:.0f}<br>時間: %{customdata:.2f}秒<extra></extra>'
                    )
                )
        
        frames.append(go.Frame(data=trace_data))
    
    # 初期データの追加
    for i, (key, df) in enumerate(dataframes.items(), 1):
        color = 'blue' if i == 1 else 'red'
        if recent_only:
            # 空のデータを表示
            fig.add_trace(go.Scatter(
                x=[],
                y=[],
                mode='lines',
                name=filenames[key],
                line=dict(color=color, width=1),
                hovertemplate='X: %{x:.0f}<br>Y: %{y:.0f}<br>時間: %{customdata:.2f}秒<extra></extra>'
            ))
            # 最新の点用のトレース
            fig.add_trace(go.Scatter(
                x=[],
                y=[],
                mode='markers',
                name=f'{filenames[key]} (現在位置)',
                marker=dict(size=8, color=color),
                hovertemplate='X: %{x:.0f}<br>Y: %{y:.0f}<br>時間: %{customdata:.2f}秒<extra></extra>'
            ))
        else:
            # 全データを表示
            mask = df['TS'] <= df['TS'].max()
            fig.add_trace(go.Scatter(
                x=df[mask]['LX'],
                y=df[mask]['LY'],
                mode='lines',
                name=filenames[key],
                line=dict(color=color, width=1),
                customdata=df[mask]['TS'],
                hovertemplate='X: %{x:.0f}<br>Y: %{y:.0f}<br>時間: %{customdata:.2f}秒<extra></extra>'
            ))
    
    # レイアウトの設定
    fig.update_layout(
        title='視線移動の比較',
        xaxis_title='X座標 (ピクセル)',
        yaxis_title='Y座標 (ピクセル)',
        showlegend=True,
        width=800,  # さらに幅を拡大
        height=400,  # さらに高さを拡大（1:2の比率を維持）
        xaxis=dict(range=[0, 4320], constrain='domain'),
        yaxis=dict(
            range=[0, 2160],
            scaleanchor="x",
            scaleratio=1.0,
            constrain='domain'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        updatemenus=[{
            'type': 'buttons',
            'showactive': True,
            'x': 1.2,  # タイトルの右側に配置
            'y': 1.1,  # タイトルと同じ高さに配置
            'buttons': [{
                'label': '再生',
                'method': 'animate',
                'args': [None, {'frame': {'duration': 50, 'redraw': True}, 'fromcurrent': True}]
            }, {
                'label': '一時停止',
                'method': 'animate',
                'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}]
            }]
        }]
    )
    
    # 円を追加
    circle_radius = 324 / 2  # 直径から半径を計算
    fig.add_shape(
        type="circle",
        xref="x",
        yref="y",
        x0=2160 - circle_radius,
        y0=1080 - circle_radius,
        x1=2160 + circle_radius,
        y1=1080 + circle_radius,
        line_color="rgba(0,0,0,0.5)",
        line_width=2,
        fillcolor="rgba(0,0,0,0)"
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGrey')
    
    # framesをFigureオブジェクトの直接のプロパティとして設定
    fig.frames = frames
    
    # スライダーの設定
    steps = []
    for t in time_steps:
        step_data = []
        for i, (key, df) in enumerate(dataframes.items(), 1):
            if recent_only:
                mask = (df['TS'] <= t) & (df['TS'] > max(t - 40, 0))
                latest_point = df[df['TS'] <= t].iloc[-1:] if len(df[df['TS'] <= t]) > 0 else pd.DataFrame()
                
                # 軌跡のデータ
                step_data.append(dict(
                    x=df[mask]['LX'].tolist(),
                    y=df[mask]['LY'].tolist()
                ))
                # 現在位置のデータ
                if not latest_point.empty:
                    step_data.append(dict(
                        x=[latest_point['LX'].iloc[0]],
                        y=[latest_point['LY'].iloc[0]]
                    ))
                else:
                    step_data.append(dict(x=[], y=[]))
            else:
                mask = df['TS'] <= t
                step_data.append(dict(
                    x=df[mask]['LX'].tolist(),
                    y=df[mask]['LY'].tolist()
                ))
        
        steps.append(
            dict(
                method='animate',
                label=f'{t:.1f}秒',
                args=[
                    [{'x': [d['x'] for d in step_data], 'y': [d['y'] for d in step_data]}],
                    {'frame': {'duration': 50, 'redraw': True}, 'mode': 'immediate'}
                ]
            )
        )
    
    fig.update_layout(
        sliders=[{
            'currentvalue': {'prefix': '時間: ', 'suffix': '秒'},
            'steps': steps,
            'transition': {'duration': 0}
        }]
    )
    
    return fig

def create_animation_frames(df, color):
    frames = []
    total_time = df['TS'].max() - df['TS'].min()
    num_frames = 50
    time_step = total_time / num_frames
    
    for i in range(num_frames + 1):
        current_time = df['TS'].min() + i * time_step
        mask = df['TS'] <= current_time
        frame_data = df[mask]
        
        frame = go.Frame(
            data=[go.Scatter(
                x=frame_data['LX'],
                y=frame_data['LY'],
                mode='lines+markers',
                line=dict(color=color),
                name=f'Time: {current_time:.2f}s'
            )],
            name=f'frame{i}'
        )
        frames.append(frame)
    
    return frames

def create_animation_video(fig, output_path, width=800, height=600, fps=2.5):
    # フレームを生成
    frames = []
    for frame in fig.frames:
        fig.update(data=frame.data)
        img_bytes = fig.to_image(format="png", width=width, height=height)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frames.append(img)

    # ビデオライターを設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # フレームを書き込み
    for frame in frames:
        out.write(frame)

    out.release()

@app.route('/download_animation/<plot_type>')
def download_animation(plot_type):
    try:
        # セッションからデータを読み込み
        temp_data = load_temp_data(session.get('session_id'))
        if not temp_data:
            raise ValueError('セッションデータが見つかりません')
        
        dataframes_dict = temp_data['dataframes']
        filenames = temp_data['filenames']
        
        # DataFrameを再構築
        dataframes = {}
        for key, df_dict in dataframes_dict.items():
            dataframes[key] = pd.DataFrame(df_dict)
        
        # 一時ファイルを作成
        temp_file = NamedTemporaryFile(suffix='.mp4', delete=False)
        temp_path = temp_file.name
        temp_file.close()

        # プロットタイプに基づいて適切なフィギュアを生成
        if plot_type == 'combined':
            fig = create_combined_plot(dataframes, filenames, False)
        elif plot_type == 'combined_right':
            fig = create_combined_plot(dataframes, filenames, True)
        elif plot_type == 'plot1':
            fig = create_gaze_plot(dataframes['データ1'], filenames['データ1'], 'blue', return_fig=True)
        elif plot_type == 'plot2':
            fig = create_gaze_plot(dataframes['データ2'], filenames['データ2'], 'red', return_fig=True)
        elif plot_type == 'plot1_right':
            fig = create_gaze_plot(dataframes['データ1'], filenames['データ1'], 'blue', recent_only=True, return_fig=True)
        elif plot_type == 'plot2_right':
            fig = create_gaze_plot(dataframes['データ2'], filenames['データ2'], 'red', recent_only=True, return_fig=True)
        else:
            raise ValueError(f'無効なプロットタイプ: {plot_type}')

        app.logger.debug(f'ビデオ生成開始: {temp_path}')
        create_animation_video(fig, temp_path)
        app.logger.debug('ビデオ生成完了')

        try:
            return send_file(
                temp_path,
                mimetype='video/mp4',
                as_attachment=True,
                download_name=f'gaze_animation_{plot_type}.mp4'
            )
        finally:
            # send_file完了後に一時ファイルを削除
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                app.logger.debug(f'一時ファイルを削除しました: {temp_path}')

    except Exception as e:
        app.logger.error(f'エラーが発生しました: {str(e)}', exc_info=True)
        return f'エラーが発生しました: {str(e)}', 500

if __name__ == '__main__':
    app.run(debug=True)
