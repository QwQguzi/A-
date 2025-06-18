import subprocess
import sys
import os
os.environ['TK_SILENCE_DEPRECATION'] = '1'  # 屏蔽 macOS 下 Tkinter 警告
import time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from requests.exceptions import ConnectTimeout, RequestException
import threading
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import pandas as pd
import queue
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

update_queue = queue.Queue()

def restart_program():
    python = sys.executable
    os.execl(python, python, *sys.argv)

def is_uptrend_narrow_channel(df_hist, window=20, max_residual_ratio=0.015):
    if len(df_hist) < window:
        return False

    recent = df_hist['收盘'].tail(window).values
    x = np.arange(window).reshape(-1, 1)
    y = recent

    model = LinearRegression()
    model.fit(x, y)
    slope = model.coef_[0]
    y_pred = model.predict(x)
    residuals = np.abs(y - y_pred)

    avg_price = np.mean(y)
    max_residual = np.max(residuals)

    # 条件：趋势向上 & 残差不超过阈值
    return slope > 0 and (max_residual / avg_price) < max_residual_ratio


# ---------------- 安装依赖函数 ----------------
def install(pkg):
    print(f"\nInstalling {pkg}...\n")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", pkg], check=True)
        print(f"\n✅ {pkg} installation complete\n")
    except subprocess.CalledProcessError:
        print(f"\n❌ Failed to install {pkg}, please check your network or permissions.\n")

def check_and_install_packages():
    packages = ['akshare', 'pandas', 'requests', 'tqdm', 'scikit-learn', 'numpy']
    for pkg in packages:
        try:
            __import__(pkg)
            print(f"✅ Package installed: {pkg}")
        except ImportError:
            print(f"⚠️ Missing package: {pkg}")
            install(pkg)





# ---------------- 核心筛选逻辑 ----------------
def stock_filter_process(min_turnover, max_turnover, code_prefixes, use_ma5, use_cache, volatility_filter, output_func, use_uptrend_channel=False, use_volatility_ratio_filter=False):

    import akshare as ak

    RETRY_COUNT = 2
    RETRY_DELAY = 0
    MAX_WORKERS = 50

    today = datetime.today()
    one_week_ago = today - timedelta(weeks=1)
    two_weeks_ago = today - timedelta(weeks=2)
    CACHE_FILE = "行情缓存.csv"

    def fetch_all(use_cache=True):
        if use_cache and os.path.exists(CACHE_FILE):
            output_func("加载缓存数据中...\n")
            try:
                df = pd.read_csv(CACHE_FILE, encoding='utf-8-sig')
                df['代码'] = df['代码'].apply(lambda x:(6 - len(str(x))) * "0" + str(x)) 
            except UnicodeDecodeError:
                df = pd.read_csv(CACHE_FILE, encoding='utf-8')
            output_func("缓存数据加载完成！\n")
            return df
        for attempt in range(RETRY_COUNT):
            try:
                output_func("请求股票数据...\n")
                df = ak.stock_zh_a_spot_em()
                if not df.empty:
                    df.to_csv(CACHE_FILE, index=False, encoding='utf-8-sig')
                    output_func("股票数据获取成功！\n")
                return df
            except (ConnectTimeout, RequestException):
                output_func(f"请求失败，重试 {attempt+1}...\n")
                time.sleep(RETRY_DELAY)
        output_func("获取股票数据失败，返回空数据\n")
        return pd.DataFrame()

    all_stocks = fetch_all(use_cache=use_cache)

    if all_stocks.empty:
        output_func("没有获取到股票数据，退出。\n")
        return

    all_stocks['代码'] = all_stocks['代码'].astype(str)
    all_stocks['换手率'] = pd.to_numeric(all_stocks['换手率'], errors='coerce')

    filtered = all_stocks.sort_values(by='换手率', ascending=False)
    filtered = filtered.loc[
        lambda df: df['代码'].astype(str).str.startswith(tuple(code_prefixes)) &
                   (df['换手率'] >= min_turnover) &
                   (df['换手率'] <= max_turnover if max_turnover is not None else True)
    ]

    if filtered.empty:
        output_func("没有符合条件的股票。\n")
        return

    def fetch_hist(symbol):
        for attempt in range(RETRY_COUNT):
            try:
                df = ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date=two_weeks_ago.strftime('%Y%m%d'),
                    end_date=today.strftime('%Y%m%d')
                )
                return df if not df.empty else None
            except (ConnectTimeout, RequestException):
                if attempt < RETRY_COUNT - 1:
                    time.sleep(RETRY_DELAY)
                else:
                    return None
        return None

    def process_stock(row, check_ma5=False):
        code, name, turnover = row['代码'], row['名称'], row['换手率']
        hist = fetch_hist(code)
        if hist is None:
            return None

        hist = hist.sort_values(by='日期')
        hist['收盘'] = pd.to_numeric(hist['收盘'], errors='coerce')
        hist['开盘'] = pd.to_numeric(hist['开盘'], errors='coerce')
        hist['成交量'] = pd.to_numeric(hist['成交量'], errors='coerce')

        hist['收盘涨幅'] = hist['收盘'].pct_change()
        recent_limit_check = hist['收盘涨幅'].iloc[-3:]
        if (recent_limit_check >= 0.198).any():
            return None


        hist['日期'] = pd.to_datetime(hist['日期'])
        hist.set_index('日期', inplace=True)

        recent = hist.loc[one_week_ago:today].copy()
        if recent.empty:
            return None

        change = (recent['收盘'] - recent['收盘'].shift(1)) / recent['收盘'].shift(1) * 100
        if change.tail(5).sum() < -20:
            return None

        if volatility_filter:
        # 其他成交量稳定条件...
            if use_volatility_ratio_filter:
                curr_avg = recent['成交量'].mean()
                prev = hist.loc[(one_week_ago - timedelta(weeks=1)):one_week_ago]
                if prev.empty or curr_avg < 2 * prev['成交量'].mean():
                    return None

        if len(recent) >= 2:
            today_vol = recent.iloc[-1]['成交量']
            yesterday_vol = recent.iloc[-2]['成交量']
            if pd.notna(today_vol) and pd.notna(yesterday_vol):
                if not (0.7 * yesterday_vol <= today_vol < 1.5 * yesterday_vol):
                    return None
        else:
            return None
        
        if use_uptrend_channel:
            if not is_uptrend_narrow_channel(hist):
                return None

        if len(recent) >= 3:
            day_before_vol = recent.iloc[-3]['成交量']
            if pd.notna(yesterday_vol) and pd.notna(day_before_vol):
                if not (0.7 * day_before_vol <= yesterday_vol < 1.5 * day_before_vol):
                    return None
        else:
            return None


        if len(recent) >= 2:
            today_close = recent.iloc[-1]['收盘']
            yesterday_close = recent.iloc[-2]['收盘']
            if pd.notna(today_close) and pd.notna(yesterday_close):
                drop_pct = (today_close - yesterday_close) / yesterday_close * 100
                if drop_pct < -7:
                    return None
        else:
            return None

        if check_ma5:
            hist['MA5'] = hist['收盘'].rolling(window=5).mean()
            recent_ma5 = hist['MA5'].dropna().tail(5)
            if len(recent_ma5) < 5 or not all(x < y for x, y in zip(recent_ma5, recent_ma5[1:])):
                return None

        recent_2w = hist.loc[two_weeks_ago:today].reset_index()

        return {
            '代码': code,
            '名称': name,
            '换手率': turnover,
            'recent_2w_df': recent_2w,  
        }


    rows = [row for _, row in filtered.iterrows()]
    output_func(f"开始处理{len(rows)}只股票，请稍等...\n")

    valid = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for r in rows:
            futures.append(executor.submit(process_stock, r, check_ma5=use_ma5))

        total = len(futures)
        for i, future in enumerate(futures, 1):
            result = future.result()
            if result:
                valid.append(result)
               
                output_func(f"符合条件：{result['代码']} {result['名称']}\n")
            else:
                output_func(f"已处理 {i}/{total} 只股票...\n")

    if valid:
        output_func("\n✅ 符合条件的股票:\n")
        
        for item in valid:
            output_func(f"{item['代码']} {item['名称']}\n")
    else:
        output_func("没有符合条件的股票。\n")


# ---------------- Tkinter GUI ----------------
def start_filter(params, text_widget, btn):
    def task():
        btn.config(state=tk.DISABLED)
        def output_func(msg,df=None):
            text_widget.configure(state='normal')
            text_widget.insert(tk.END, msg)
            text_widget.see(tk.END)
            text_widget.configure(state='disabled')
        try:
            stock_filter_process(**params, output_func=output_func)
        except Exception as e:
            output_func(f"\n错误:{e}\n")
        btn.config(state=tk.NORMAL)
    threading.Thread(target=task, daemon=True).start()

def run_gui():
    root = tk.Tk()
    root.title("股票筛选工具")
    root.geometry("700x500")

    frm = ttk.Frame(root, padding=10)
    frm.pack(fill=tk.BOTH, expand=True)

    input_frame = ttk.LabelFrame(frm, text="筛选参数", padding=10)
    input_frame.pack(fill=tk.X)

    btn_frame = ttk.Frame(frm)
    btn_frame.pack(fill=tk.X, pady=5)

    ttk.Label(input_frame, text="最低换手率:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
    min_turnover_var = tk.DoubleVar(value=10)
    ttk.Entry(input_frame, textvariable=min_turnover_var, width=10).grid(row=0, column=1, sticky=tk.W)

    ttk.Label(input_frame, text="最高换手率:").grid(row=0, column=2, sticky=tk.W, padx=5, pady=5)
    max_turnover_var = tk.DoubleVar(value=100)
    ttk.Entry(input_frame, textvariable=max_turnover_var, width=10).grid(row=0, column=3, sticky=tk.W)

    ttk.Label(input_frame, text="代码前缀(逗号分隔):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
    code_prefix_var = tk.StringVar(value="30,60,00")
    ttk.Entry(input_frame, textvariable=code_prefix_var, width=30).grid(row=1, column=1, columnspan=3, sticky=tk.W)

    use_ma5_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(input_frame, text="使用5日均线过滤", variable=use_ma5_var).grid(row=2, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)

    use_cache_var = tk.BooleanVar(value=True)
    ttk.Checkbutton(input_frame, text="是否使用缓存", variable=use_cache_var).grid(row=2, column=2, columnspan=2, sticky=tk.W, padx=5, pady=5)

    volatility_filter_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(input_frame, text="是否筛选成交量稳定", variable=volatility_filter_var).grid(row=2, column=4, columnspan=2, sticky="w", padx=5, pady=5)

    uptrend_channel_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(input_frame, text="是否开启上涨窄通道筛选", variable=uptrend_channel_var).grid(row=3, column=0, columnspan=3, sticky=tk.W, padx=5, pady=5)

    use_volatility_ratio_filter_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(input_frame, text="开启成交量倍数筛选(2倍)", variable=use_volatility_ratio_filter_var).grid(row=4, column=0, columnspan=3, sticky=tk.W, padx=5, pady=5)

    restart_btn = ttk.Button(btn_frame, text="重启程序", command=restart_program)
    restart_btn.pack(side=tk.LEFT, padx=5)


    output_text = scrolledtext.ScrolledText(frm, height=20, state='disabled', wrap=tk.WORD)
    output_text.pack(fill=tk.BOTH, expand=True)



    def on_start():
        try:
            min_turnover = min_turnover_var.get()
            max_turnover = max_turnover_var.get()
            code_prefixes = [p.strip() for p in code_prefix_var.get().split(',') if p.strip()]
            use_ma5 = use_ma5_var.get()

            if min_turnover < 0 or (max_turnover is not None and max_turnover < min_turnover):
                messagebox.showerror("参数错误", "请确保换手率数值合理，最高换手率不小于最低换手率。")
                return
            if not code_prefixes:
                messagebox.showerror("参数错误", "请输入至少一个代码前缀。")
                return

            params = {
                'min_turnover': min_turnover,
                'max_turnover': max_turnover,
                'code_prefixes': code_prefixes,
                'use_ma5': use_ma5,
                'use_cache': use_cache_var.get(),
                'volatility_filter': volatility_filter_var.get(),
                'use_uptrend_channel': uptrend_channel_var.get(),
                'use_volatility_ratio_filter': use_volatility_ratio_filter_var.get(),

            }
            output_text.configure(state='normal')
            output_text.delete('1.0', tk.END)
            output_text.configure(state='disabled')
            start_filter(params, output_text, start_btn)
        except Exception as e:
            messagebox.showerror("错误", str(e))
    

    
    def output_func(msg):
        update_queue.put((msg))



    def process_queue():
        try:
            while True:
                msg, df = update_queue.get_nowait()
                if msg.startswith("符合条件："):
                    output_text.configure(state='normal')
                    output_text.insert(tk.END, msg)
                    output_text.see(tk.END)
                    output_text.configure(state='disabled')
                else:
                   
                    output_text.configure(state='normal')
                    output_text.insert(tk.END, msg)
                    output_text.see(tk.END)
                    output_text.configure(state='disabled')
        except queue.Empty:
            pass
        root.after(300, process_queue)

   
    process_queue()

    start_btn = ttk.Button(btn_frame, text="开始筛选", command=on_start)
    start_btn.pack(side=tk.LEFT, padx=5)

    root.mainloop()

if __name__ == "__main__":
    check_and_install_packages()
    run_gui()