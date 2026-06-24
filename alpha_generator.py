import pandas as pd
import os
import qlib
from qlib.data import D
from qlib.constant import REG_CN
from qlib.contrib.data.handler import Alpha158

def setup_qlib_env():
    """设置qlib环境和必要的文件"""
    # 设置qlib路径（Windows环境）
    qlib_dir = os.path.expanduser("~/.qlib/qlib_data/custom_data")
    inst_dir = os.path.join(qlib_dir, "instruments")
    os.makedirs(inst_dir, exist_ok=True)

    # 初始化qlib
    qlib.init(provider_uri=qlib_dir, region=REG_CN)

    # 创建股票列表文件
    try:
        # 从all.txt读取股票列表
        with open(os.path.join(inst_dir, "all.txt"), "r") as f:
            stocks = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        # 如果all.txt不存在，从CSV数据创建
        df = pd.read_csv("data/csi300_202401.csv")
        stocks = df['stock_id'].unique().tolist()
        with open(os.path.join(inst_dir, "all.txt"), "w") as f:
            for stock in stocks:
                f.write(f"{stock}\n")

    # 创建csi300.txt
    with open(os.path.join(inst_dir, "csi300.txt"), "w") as f:
        for stock in stocks:
            f.write(f"{stock}\n")

def check_data_quality(df):
    """检查数据质量"""
    print("数据基本信息:")
    print("-" * 50)
    print("数据维度:", df.shape)
    print("总记录数:", len(df))
    print("股票数量:", df['stock_id'].nunique())
    print("\n列名:", list(df.columns))
    
    print("\n时间范围:")
    print("起始日期:", df['日期'].min())
    print("结束日期:", df['日期'].max())
    print("交易日数量:", df['日期'].nunique())
    
    print("\n缺失值统计:")
    print(df.isnull().sum())

def prepare_qlib_data(input_file="data/csi300_202401.csv", csv_dir="csv_data"):
    """准备qlib格式数据"""
    # 创建目录
    os.makedirs(csv_dir, exist_ok=True)
    
    # 读取数据
    df = pd.read_csv(input_file)
    check_data_quality(df)
    
    # 数据处理
    df['vwap'] = df['成交额'] / df['成交量']
    df = df.rename(columns={
        '开盘': 'open', '收盘': 'close', '最高': 'high',
        '最低': 'low', '成交量': 'volume', '日期': 'date'
    })
    df['date'] = pd.to_datetime(df['date'])
    
    # 分股票存储
    for stock in df['stock_id'].unique():
        stock_df = df[df['stock_id']==stock][['date','open','close','high','low','volume','vwap']]
        file_path = os.path.join(csv_dir, f"{stock}.csv")
        stock_df.to_csv(file_path, index=False)

def convert_to_qlib_format():
    """将CSV数据转换为qlib格式"""
    try:
        # 创建必要的目录
        qlib_dir = os.path.expanduser("~/.qlib/qlib_data/custom_data")
        os.makedirs(qlib_dir, exist_ok=True)
        os.makedirs("csv_data", exist_ok=True)

        # 调用dump_bin.py进行转换
        cmd = (
            "python qlib/scripts/dump_bin.py dump_all "
            "--csv_path ./csv_data "
            f"--qlib_dir {qlib_dir} "
            "--include_fields open,close,high,low,volume,vwap "
            "--date_field_name date"
        )
        
        print("\n开始转换数据为qlib格式...")
        import subprocess
        subprocess.run(cmd, shell=True, check=True)
        print("数据转换完成!")
        
    except Exception as e:
        print(f"数据转换失败: {str(e)}")
        raise       

def generate_alpha158(start_time="20080101", end_time="20201231", output_file="data/alpha158_features.pkl"):
    """生成Alpha158特征"""
    # 首先设置环境
    setup_qlib_env()
    
    # 创建Alpha158处理器
    handler = Alpha158(
        instruments="csi300",
        start_time=start_time,
        end_time=end_time,
        freq="day",
    )
    
    # 设置标签配置
    handler_config = {
        "label": ["Ref($close, -5) / Ref($close, -1) - 1"]
    }
    handler.config = handler_config
    
    # 获取并保存数据
    df = handler.fetch()
    df.to_pickle(output_file)
    print(f"Alpha158特征已保存到: {output_file}")
    return df

def main():
   try:
       # 1. 检查输入数据
       input_file = "data/csi300_202401.csv"
       if not os.path.exists(input_file):
           raise FileNotFoundError(f"未找到输入文件: {input_file}")
       
       # 2. 准备qlib格式数据
       print("\n开始准备qlib格式数据...")
       prepare_qlib_data()
       print("qlib格式数据准备完成!")

       convert_to_qlib_format()
       
       # 3. 检查必要的目录
       qlib_dir = os.path.expanduser("~/.qlib/qlib_data/custom_data")
       if not os.path.exists(qlib_dir):
           raise RuntimeError(f"qlib数据目录不存在: {qlib_dir}")
           
       # 4. 生成Alpha158特征
       print("\n开始生成Alpha158特征...")
       alpha_df = generate_alpha158()
       
       # 5. 检查生成的特征
       if alpha_df is not None:
           print("\nAlpha158特征生成成功:")
           print(f"特征维度: {alpha_df.shape}")
           print(f"时间范围: {alpha_df.index.get_level_values('datetime').min()} 到 {alpha_df.index.get_level_values('datetime').max()}")
           print(f"股票数量: {len(alpha_df.index.get_level_values('instrument').unique())}")
       else:
           raise RuntimeError("Alpha158特征生成失败")

   except Exception as e:
       print(f"\n错误: {str(e)}")
       raise

if __name__ == "__main__":
   main()