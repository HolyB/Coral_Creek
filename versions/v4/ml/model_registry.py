"""
模型注册中心 - Hugging Face Hub 集成
Model Registry - HuggingFace Hub Integration

功能:
- 上传模型到 HuggingFace Hub
- 下载模型用于推理
- 版本管理
"""

import os
import json
import joblib
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# HuggingFace Hub
try:
    from huggingface_hub import HfApi, hf_hub_download, upload_file
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("⚠️ huggingface_hub 未安装，模型将只保存到本地")


class ModelRegistry:
    """模型注册中心"""
    
    def __init__(self, 
                 repo_id: str = "coral-creek-models",
                 local_dir: str = None):
        """
        Args:
            repo_id: HuggingFace repo 名称 (会自动加上用户名前缀)
            local_dir: 本地模型目录
        """
        self.repo_id = repo_id
        self.local_dir = Path(local_dir or os.path.dirname(__file__)) / "saved_models"
        self.local_dir.mkdir(parents=True, exist_ok=True)
        
        self.api = HfApi() if HF_AVAILABLE else None
        self._full_repo_id = None
    
    @property
    def full_repo_id(self) -> str:
        """获取完整的 repo ID (包含用户名)"""
        if self._full_repo_id is None and self.api:
            try:
                user = self.api.whoami()
                self._full_repo_id = f"{user['name']}/{self.repo_id}"
            except:
                self._full_repo_id = self.repo_id
        return self._full_repo_id or self.repo_id
    
    def save_local(self, 
                   model: Any, 
                   model_name: str,
                   metadata: Dict = None) -> Path:
        """
        保存模型到本地
        
        Args:
            model: 模型对象 (XGBoost, sklearn, etc.)
            model_name: 模型名称 (如 'xgb_signal_predictor')
            metadata: 模型元数据 (训练参数、指标等)
        
        Returns:
            保存路径
        """
        # 创建模型目录
        model_dir = self.local_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        model_path = model_dir / "model.joblib"
        joblib.dump(model, model_path)
        
        # 保存元数据
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'saved_at': datetime.now().isoformat(),
            'model_name': model_name,
        })
        
        meta_path = model_dir / "metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f"✅ 模型已保存到本地: {model_path}")
        return model_path
    
    def load_local(self, model_name: str) -> tuple:
        """
        从本地加载模型
        
        Returns:
            (model, metadata)
        """
        model_dir = self.local_dir / model_name
        model_path = model_dir / "model.joblib"
        meta_path = model_dir / "metadata.json"
        
        if not model_path.exists():
            raise FileNotFoundError(f"模型不存在: {model_path}")
        
        model = joblib.load(model_path)
        
        metadata = {}
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)
        
        return model, metadata
    
    def upload_to_hub(self, 
                      model_name: str,
                      commit_message: str = None) -> str:
        """
        上传模型到 HuggingFace Hub
        
        Args:
            model_name: 本地模型名称
            commit_message: 提交信息
        
        Returns:
            HuggingFace URL
        """
        if not HF_AVAILABLE or not self.api:
            print("❌ HuggingFace Hub 不可用")
            return None
        
        model_dir = self.local_dir / model_name
        
        if not model_dir.exists():
            raise FileNotFoundError(f"本地模型不存在: {model_dir}")
        
        # 确保 repo 存在
        try:
            self.api.create_repo(
                repo_id=self.full_repo_id,
                repo_type="model",
                exist_ok=True,
                private=True  # 私有仓库
            )
        except Exception as e:
            print(f"⚠️ 创建/检查 repo: {e}")
        
        # 上传文件
        commit_message = commit_message or f"Update {model_name} - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        for file_path in model_dir.glob("*"):
            if file_path.is_file():
                self.api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=f"{model_name}/{file_path.name}",
                    repo_id=self.full_repo_id,
                    commit_message=commit_message
                )
                print(f"  📤 上传: {file_path.name}")
        
        url = f"https://huggingface.co/{self.full_repo_id}"
        print(f"✅ 模型已上传到: {url}")
        return url
    
    def download_from_hub(self, 
                          model_name: str,
                          force: bool = False) -> Path:
        """
        从 HuggingFace Hub 下载模型
        
        Args:
            model_name: 模型名称
            force: 强制重新下载
        
        Returns:
            本地模型目录
        """
        if not HF_AVAILABLE:
            raise RuntimeError("HuggingFace Hub 不可用")
        
        model_dir = self.local_dir / model_name
        model_path = model_dir / "model.joblib"
        
        # 如果本地已存在且不强制下载
        if model_path.exists() and not force:
            print(f"📦 使用本地缓存: {model_path}")
            return model_dir
        
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 下载文件
        for filename in ["model.joblib", "metadata.json"]:
            try:
                downloaded_path = hf_hub_download(
                    repo_id=self.full_repo_id,
                    filename=f"{model_name}/{filename}",
                    local_dir=self.local_dir,
                    local_dir_use_symlinks=False
                )
                print(f"  📥 下载: {filename}")
            except Exception as e:
                if filename == "model.joblib":
                    raise
                print(f"  ⚠️ 跳过: {filename} ({e})")
        
        print(f"✅ 模型已下载到: {model_dir}")
        return model_dir
    
    def list_models(self) -> Dict[str, Dict]:
        """列出所有本地模型"""
        models = {}
        
        for model_dir in self.local_dir.iterdir():
            if model_dir.is_dir():
                meta_path = model_dir / "metadata.json"
                if meta_path.exists():
                    with open(meta_path) as f:
                        models[model_dir.name] = json.load(f)
                else:
                    models[model_dir.name] = {'model_name': model_dir.name}
        
        return models


# === 便捷函数 ===

_registry = None

def get_registry() -> ModelRegistry:
    """获取全局 ModelRegistry 实例"""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


def save_model(model: Any, model_name: str, metadata: Dict = None, upload: bool = False) -> Path:
    """保存模型 (可选上传到 Hub)"""
    registry = get_registry()
    path = registry.save_local(model, model_name, metadata)
    
    if upload:
        registry.upload_to_hub(model_name)
    
    return path


def load_model(model_name: str, from_hub: bool = False):
    """加载模型 (可选从 Hub 下载)"""
    registry = get_registry()
    
    if from_hub:
        try:
            registry.download_from_hub(model_name)
        except Exception as e:
            print(f"⚠️ 从 Hub 下载失败，使用本地: {e}")
    
    return registry.load_local(model_name)


# === 测试 ===
if __name__ == "__main__":
    print("=== Model Registry 测试 ===\n")
    
    registry = ModelRegistry()
    
    # 检查 HF 登录状态
    if registry.api:
        try:
            user = registry.api.whoami()
            print(f"✅ HuggingFace 已登录: {user['name']}")
            print(f"   Repo ID: {registry.full_repo_id}")
        except Exception as e:
            print(f"❌ HuggingFace 未登录: {e}")
            print("   运行: huggingface-cli login")
    else:
        print("❌ huggingface_hub 未安装")
    
    # 列出本地模型
    print("\n本地模型:")
    models = registry.list_models()
    if models:
        for name, meta in models.items():
            print(f"  - {name}: {meta.get('saved_at', 'unknown')}")
    else:
        print("  (无)")
