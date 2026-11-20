"""资源与耗时监控：训练/推理的「占用资源」口径

为什么需要：
  只报 F1 的对比是不完整的。真实选型还要回答「训练多久、跑一条多贵、吃多少内存/显存」。
  本文件给出统一采集口径，三路方法都用同一套函数，避免「这个脚本测的是 RSS、那个测的是虚拟内存」。

口径定义：
  train_seconds   训练阶段墙钟时间（规则法为 0，因为它不训练）
  infer_ms_b1     单条推理（batch=1）中位数，代表交互式场景
  infer_ms_b32    批量推理（batch=32）均摊到单条，代表批处理吞吐
  throughput      条/秒（batch=32 口径）
  peak_rss_mb     进程峰值常驻内存（psutil 采样，0.2s 一次）
  peak_gpu_mb     训练/推理期间 GPU 峰值显存（torch.cuda.max_memory_allocated）
  model_size_mb   落盘模型文件大小（磁盘占用，含 ckpt 里的优化器无关部分）
"""

import threading
import time

import psutil


def file_size_mb(path):
    from pathlib import Path
    p = Path(path)
    if p.is_file():
        return round(p.stat().st_size / 1024 / 1024, 1)
    if p.is_dir():
        return round(sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1024 / 1024, 1)
    return 0.0


class ResourceMonitor:
    """后台线程采样 RSS + GPU 峰值显存"""

    def __init__(self, interval=0.2):
        self.interval = interval
        self.peak_rss_mb = 0.0
        self.peak_gpu_mb = 0.0
        self._stop = threading.Event()
        self._thread = None

    def _sample_gpu(self):
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                return torch.cuda.max_memory_allocated() / 1024 / 1024
        except Exception:
            pass
        return 0.0

    def _loop(self):
        proc = psutil.Process()
        while not self._stop.is_set():
            try:
                rss = proc.memory_info().rss / 1024 / 1024
                self.peak_rss_mb = max(self.peak_rss_mb, rss)
            except Exception:
                pass
            self.peak_gpu_mb = max(self.peak_gpu_mb, self._sample_gpu())
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.5)
        try:
            self.peak_rss_mb = max(self.peak_rss_mb, psutil.Process().memory_info().rss / 1024 / 1024)
        except Exception:
            pass
        self.peak_gpu_mb = max(self.peak_gpu_mb, self._sample_gpu())
        return False


class Timer:
    def __init__(self):
        self.t0 = None
        self.seconds = 0.0

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.seconds = round(time.time() - self.t0, 2)
        return False


def env_info():
    """记录跑分机器，换机器后数字不可比的提醒"""
    import platform
    import sys
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1)
        else:
            info["gpu"] = "CPU"
    except Exception:
        info["gpu"] = "unknown"
    try:
        import sklearn
        info["sklearn"] = sklearn.__version__
    except Exception:
        pass
    return info
