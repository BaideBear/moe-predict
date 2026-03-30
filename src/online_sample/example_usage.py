import torch
import time
import threading
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer

from online_sample.data_structures import ModelConfig
from online_sample.buffer import create_buffer
from online_sample.sampler import OnlineSampler
from online_sample.predictor_interface import create_predictor_interface
from online_sample.utils import extract_model_config


def print_header(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_section(title):
    print(f"\n[{title}]")


def print_config(config_dict):
    for key, value in config_dict.items():
        print(f"  {key}: {value}")


def main():
    """
    在线采样完整流程演示
    
    本脚本演示如何使用在线采样模块进行MoE模型的激活值采集，
    并在预测器训练端接收数据进行处理。
    
    流程说明：
    1. 加载Mixtral 8x7B模型和tokenizer
    2. 创建激活值缓冲区（buffer）
    3. 启动在线采样器（sampler）从MMLU训练集采集数据
    4. 启动预测器训练线程，从buffer读取数据进行处理
    5. 监控采样和训练过程
    6. 清理资源
    """
    
    print_header("在线采样完整流程演示")
    
    # ========================================
    # 1. 配置参数
    # ========================================
    
    # 模型配置
    MODEL_PATH = "/data1/gx/MoE-predict/models/Mixtral-8x7B-v0.1"
    MODEL_NAME = "Mixtral-8x7B-v0.1"
    
    # 数据集配置
    DATASET_PATH = "/data1/gx/MoE-predict/dataset/processed/train/mmlu.jsonl"
    
    # 采样配置
    PATTERN = "attn_gate"  # 支持的模式: "attn_gate", "gate_input", "token_gate"
    BATCH_SIZE = 1  # 采样时的batch size
    MAX_SEQ_LENGTH = 2048  # 最大序列长度
    
    # 缓冲区配置
    BUFFER_SIZE_GB = 4.0  # 缓冲区大小（GB），默认4GB
    DEVICE = "cuda"  # 设备类型
    
    # 预测器训练配置
    PREDICTOR_BATCH_SIZE = 1  # 预测器训练时的batch size
    PREDICTOR_TIMEOUT = None  # 读取超时时间，None表示无限等待
    TRAINING_INTERVAL = 1.0  # 训练间隔（秒），模拟训练时间
    
    # 运行配置
    MAX_SAMPLES = 20  # 最大采样数量（用于演示）
    MONITOR_INTERVAL = 2.0  # 监控间隔（秒）
    
    print_section("配置信息")
    print_config({
        "模型路径": MODEL_PATH,
        "模型名称": MODEL_NAME,
        "数据集路径": DATASET_PATH,
        "采样模式": PATTERN,
        "采样batch size": BATCH_SIZE,
        "最大序列长度": MAX_SEQ_LENGTH,
        "缓冲区大小": f"{BUFFER_SIZE_GB} GB",
        "设备": DEVICE,
        "预测器batch size": PREDICTOR_BATCH_SIZE,
        "训练间隔": f"{TRAINING_INTERVAL} 秒",
        "最大采样数量": MAX_SAMPLES
    })
    
    # ========================================
    # 2. 加载模型和tokenizer
    # ========================================
    
    print_section("步骤1: 加载模型和tokenizer")
    
    print("  正在加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager"
    )
    model.eval()
    
    print("  正在加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("  ✓ 模型和tokenizer加载完成！")
    
    # ========================================
    # 3. 提取模型配置
    # ========================================
    
    print_section("步骤2: 提取模型配置")
    
    model_config = extract_model_config(
        model=model,
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH
    )
    
    print_config({
        "模型名称": model_config.model_name,
        "层数": model_config.num_layers,
        "隐藏层维度": model_config.hidden_dim,
        "专家数量": model_config.num_experts,
        "最大序列长度": model_config.max_seq_length,
        "数据类型": model_config.dtype
    })
    
    # ========================================
    # 4. 创建激活值缓冲区
    # ========================================
    
    print_section("步骤3: 创建激活值缓冲区")
    
    buffer = create_buffer(
        model_config=model_config,
        pattern=PATTERN,
        buffer_size_gb=BUFFER_SIZE_GB,
        device=DEVICE
    )
    
    print_config({
        "模式": PATTERN,
        "大小": f"{BUFFER_SIZE_GB} GB",
        "设备": DEVICE
    })
    print("  ✓ 缓冲区创建完成！")
    
    # ========================================
    # 5. 创建在线采样器
    # ========================================
    
    print_section("步骤4: 创建在线采样器")
    
    sampler = OnlineSampler(
        model=model,
        tokenizer=tokenizer,
        dataset_path=DATASET_PATH,
        buffer=buffer,
        pattern=PATTERN,
        batch_size=BATCH_SIZE,
        max_seq_length=MAX_SEQ_LENGTH,
        trust_remote_code=True
    )
    
    print_config({
        "模式": PATTERN,
        "Batch size": BATCH_SIZE,
        "最大序列长度": MAX_SEQ_LENGTH
    })
    print("  ✓ 采样器创建完成！")
    
    # ========================================
    # 6. 创建预测器接口
    # ========================================
    
    print_section("步骤5: 创建预测器接口")
    
    predictor_interface = create_predictor_interface(
        buffer=buffer,
        pattern=PATTERN,
        batch_size=PREDICTOR_BATCH_SIZE,
        timeout=PREDICTOR_TIMEOUT
    )
    
    print_config({
        "Batch size": PREDICTOR_BATCH_SIZE,
        "超时时间": PREDICTOR_TIMEOUT
    })
    print("  ✓ 预测器接口创建完成！")
    
    # ========================================
    # 7. 预测器训练线程（模拟）
    # ========================================
    
    print_section("步骤6: 创建预测器训练线程")
    
    training_stats = {
        'total_samples_processed': 0,
        'total_batches_processed': 0,
        'start_time': time.time()
    }
    stop_training = threading.Event()
    training_lock = threading.Lock()
    
    def predictor_training_thread():
        """
        预测器训练线程
        
        这个线程模拟预测器的训练过程：
        1. 从buffer读取激活值数据
        2. 进行训练处理（这里只是模拟，实际训练代码需要实现）
        3. 重复上述过程
        """
        print("  [训练线程] 启动...")
        
        while not stop_training.is_set():
            try:
                batch = predictor_interface.get_batch()
                
                if batch is not None:
                    for data in batch:
                        with training_lock:
                            training_stats['total_samples_processed'] += 1
                            training_stats['total_batches_processed'] += 1
                            current_sample = training_stats['total_samples_processed']
                        
                        if current_sample <= 3 or current_sample % 5 == 0:
                            print(f"\n  [训练线程] 处理样本 #{current_sample}")
                            print(f"    序列长度: {data.seq_lengths.item()}")
                            print(f"    Tokens: {data.tokens.shape}")
                            print(f"    Gate logits: {data.gate_logits.shape}")
                            print(f"    Attn hidden states: {data.attn_hidden_states.shape}")
                    
                    time.sleep(TRAINING_INTERVAL)
                else:
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"  [训练线程] 错误: {e}")
                break
        
        print(f"\n  [训练线程] 停止 (已处理 {training_stats['total_samples_processed']} 样本)")
    
    training_thread = threading.Thread(target=predictor_training_thread)
    training_thread.start()
    
    # ========================================
    # 8. 启动在线采样
    # ========================================
    
    print_section("步骤7: 启动在线采样")
    
    sampler.start()
    print("  ✓ 在线采样已启动！")
    
    # ========================================
    # 9. 监控采样和训练过程
    # ========================================
    
    print_section("步骤8: 监控采样和训练过程")
    print("  (按 Ctrl+C 停止)")
    
    try:
        start_time = time.time()
        last_sample_count = 0
        last_training_count = 0
        
        while True:
            elapsed_time = time.time() - start_time
            buffer_size = buffer.get_size()
            stats = predictor_interface.get_stats()
            
            with training_lock:
                samples_processed = training_stats['total_samples_processed']
            
            sample_rate = buffer_size / elapsed_time if elapsed_time > 0 else 0
            training_rate = samples_processed / elapsed_time if elapsed_time > 0 else 0
            
            print("\n" + "-" * 80)
            print(f"  运行时间: {elapsed_time:.1f}秒")
            print(f"\n  [缓冲区]")
            print(f"    当前样本数: {buffer_size}")
            print(f"    内存使用: {stats['used_memory_gb']:.2f} GB / {stats['buffer_size_gb']:.2f} GB ({stats['utilization']*100:.1f}%)")
            print(f"    采样速率: {sample_rate:.2f} 样本/秒")
            
            print(f"\n  [训练]")
            print(f"    已处理样本: {samples_processed} / {MAX_SAMPLES}")
            print(f"    训练速率: {training_rate:.2f} 样本/秒")
            
            progress = (samples_processed / MAX_SAMPLES) * 100
            bar_length = 40
            filled = int(bar_length * samples_processed / MAX_SAMPLES)
            bar = '█' * filled + '░' * (bar_length - filled)
            print(f"    进度: [{bar}] {progress:.1f}%")
            
            if samples_processed >= MAX_SAMPLES:
                print(f"\n  ✓ 已达到目标样本数 {MAX_SAMPLES}")
                break
            
            time.sleep(MONITOR_INTERVAL)
            
    except KeyboardInterrupt:
        print("\n\n  收到停止信号...")
    
    # ========================================
    # 10. 停止采样和训练
    # ========================================
    
    print_section("步骤9: 停止采样和训练")
    
    print("  停止采样器...")
    sampler.stop()
    
    print("  停止训练线程...")
    stop_training.set()
    training_thread.join(timeout=10.0)
    
    # ========================================
    # 11. 清理资源
    # ========================================
    
    print_section("步骤10: 清理资源")
    
    buffer.stop()
    
    del model
    del tokenizer
    torch.cuda.empty_cache()
    
    print("  ✓ 资源清理完成！")
    
    # ========================================
    # 12. 最终统计
    # ========================================
    
    final_stats = predictor_interface.get_stats()
    elapsed_time = time.time() - start_time
    
    print_header("运行完成")
    
    print_section("最终统计")
    print_config({
        "总运行时间": f"{elapsed_time:.2f} 秒",
        "总采样样本数": final_stats['total_samples'],
        "总处理样本数": training_stats['total_samples_processed'],
        "平均采样速率": f"{final_stats['total_samples']/elapsed_time:.2f} 样本/秒",
        "平均训练速率": f"{training_stats['total_samples_processed']/elapsed_time:.2f} 样本/秒",
        "缓冲区峰值利用率": f"{final_stats['utilization']*100:.1f}%"
    })
    
    print("\n" + "=" * 80)
    print("  演示结束！")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
