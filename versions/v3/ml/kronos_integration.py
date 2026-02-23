import os
import sys
import pandas as pd
import importlib
import importlib.util
from typing import Optional

# 动态添加 Kronos 核心源码路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
KRONOS_DIR = os.path.join(CURRENT_DIR, "Kronos")
KRONOS_MODEL_DIR = os.path.join(KRONOS_DIR, "model")

def _force_import_kronos():
    """
    用 importlib 从绝对路径强制加载 Kronos 的 model 包。
    这比 sys.path.insert 更可靠,避免 CI 环境里的路径歧义问题。
    """
    model_init = os.path.join(KRONOS_MODEL_DIR, "__init__.py")
    if not os.path.exists(model_init):
        raise ImportError(f"Kronos model package not found at {model_init}")
    
    # 先确保 KRONOS_DIR 在 sys.path 最前面 (model 内部有相对导入依赖)
    if KRONOS_DIR not in sys.path:
        sys.path.insert(0, KRONOS_DIR)
    
    # 如果 'model' 已经被其他地方加载了,先清掉
    for key in list(sys.modules.keys()):
        if key == 'model' or key.startswith('model.'):
            del sys.modules[key]
    
    # 用 importlib 从绝对路径加载
    spec = importlib.util.spec_from_file_location(
        "model",
        model_init,
        submodule_search_locations=[KRONOS_MODEL_DIR]
    )
    model_mod = importlib.util.module_from_spec(spec)
    sys.modules["model"] = model_mod
    spec.loader.exec_module(model_mod)
    
    return model_mod

try:
    _model = _force_import_kronos()
    Kronos = _model.Kronos
    KronosTokenizer = _model.KronosTokenizer
    KronosPredictor = _model.KronosPredictor
    print(f"✅ Kronos model loaded from {KRONOS_MODEL_DIR}")
except Exception as e:
    print(f"Failed to import Kronos components: {e}")
    import traceback
    traceback.print_exc()
    Kronos = KronosTokenizer = KronosPredictor = None

class KronosEngine:
    """
    Kronos 大模型单例加载器与预测引擎。
    管理模型生命周期，避免重复加载几十MB/上百MB的权重参数。
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(KronosEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
        
    def __init__(self):
        if self._initialized:
            return
            
        print("🚀 [KronosEngine] Initializing...")
        if Kronos is None:
            raise RuntimeError("Kronos modules not found. Ensure 'versions/v3/ml/Kronos' exists.")
            
        # 设置模型规模 (可选: NeoQuasar/Kronos-small, NeoQuasar/Kronos-mini)
        self.model_name = "NeoQuasar/Kronos-small"
        self.tokenizer_name = "NeoQuasar/Kronos-Tokenizer-base"
        
        self.tokenizer = None
        self.model = None
        self.predictor = None
        
        self._load_models()
        self._initialized = True
        print("✅ [KronosEngine] Initialization absolute complete!")
        
    def _load_models(self):
        print(f"📦 Loading Tokenizer: {self.tokenizer_name}")
        self.tokenizer = KronosTokenizer.from_pretrained(self.tokenizer_name)
        
        print(f"📦 Loading Main Model: {self.model_name}")
        self.model = Kronos.from_pretrained(self.model_name)
        import torch
        # 限制 PyTorch 线程并发数并强制 CPU
        # macOS MPS 模块在 Streamlit 的多线程环境和自回归循环下易发发底层死锁 (Metal kernel trap)
        # 用纯 CPU 算力在 Mac 端运行 ~99M 小网络反而更稳、更快
        torch.set_num_threads(4) 
        
        # 实例化预测器
        # max_context 控制最大输入 K 线长度 (推荐 512, 因为 token 并不仅仅是一维的)
        self.predictor = KronosPredictor(self.model, self.tokenizer, device="cpu", max_context=512)
        
    def predict_future_klines(
        self, 
        history_df: pd.DataFrame, 
        pred_len: int = 20, 
        temperature: float = 1.0, 
        top_p: float = 0.9
    ) -> Optional[pd.DataFrame]:
        """
        基于历史 K 线生成未来的 K 线预测。
        
        :param history_df: 需要包含 ['open', 'high', 'low', 'close', 'volume'] 和 index/timestamps
        :param pred_len: 想要预测的未来 K 线根数
        :param temperature: 随机性 (0.1 ~ 1.0)
        :param top_p: 采样范围
        :return: 预测量价 DataFrame, index 为预测出来的时间戳序列
        """
        try:
            # 严格筛选模型所需的标准特征列
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            df_slice = history_df[required_cols].copy()
            
            # Kronos 原生样例需要将 amount 置入，如果没有用 volume 凑数
            if 'amount' not in history_df.columns:
                df_slice['amount'] = history_df['close'] * history_df['volume']
            else:
                df_slice['amount'] = history_df['amount']
                
            # 提取历史时间戳列
            if 'timestamps' in history_df.columns:
                x_timestamp = pd.to_datetime(history_df['timestamps'])
            elif isinstance(history_df.index, pd.DatetimeIndex):
                x_timestamp = pd.Series(history_df.index)
            else:
                x_timestamp = pd.Series(pd.date_range(end=pd.Timestamp.today(), periods=len(df_slice), freq='B'))
                
            x_timestamp = x_timestamp.reset_index(drop=True)
            df_slice = df_slice.reset_index(drop=True)
            
            # 生成未来时间戳锚点 (按照 B: 工作日 生成序列)
            last_date = x_timestamp.iloc[-1]
            y_timestamp = pd.Series(pd.date_range(start=last_date + pd.Timedelta(days=1), periods=pred_len, freq='B'))
            
            # 运行模型推断
            _pred_df = self.predictor.predict(
                df=df_slice,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=pred_len,
                T=temperature,
                top_p=top_p,
                sample_count=1,
                verbose=False
            )
            
            # 把预测时间的特征列重置成目标结构
            _pred_df.index = y_timestamp.values
            _pred_df.index.name = "date"
            
            # 统一首写大写格式输出，无缝兼容 Coral Creek 原本的图形组件
            _pred_df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }, inplace=True)
            
            return _pred_df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
        except Exception as e:
            print(f"❌ [KronosEngine] Prediction failed: {e}")
            import traceback
            traceback.print_exc()
            return None

# 单例暴露
kronos_engine = None

def get_kronos_engine() -> KronosEngine:
    global kronos_engine
    if kronos_engine is None:
        kronos_engine = KronosEngine()
    return kronos_engine
