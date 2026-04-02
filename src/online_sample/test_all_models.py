import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer

from online_sample.data_structures import ModelConfig
from online_sample.buffer import create_buffer
from online_sample.sampler import OnlineSampler
from online_sample.utils import extract_model_config


def test_model(model_path, model_name, dataset_path, max_samples=5, timeout=300, use_flash_attn=False):
    """
    测试指定模型能否正常进行 online sample
    
    Args:
        model_path: 模型路径
        model_name: 模型名称
        dataset_path: 数据集路径
        max_samples: 最大采样数量
        timeout: 超时时间（秒）
        use_flash_attn: 是否使用 FlashAttention
    """
    print("=" * 80)
    print(f"测试模型: {model_name}")
    print("=" * 80)
    
    try:
        # 1. 加载模型
        print(f"\n[1/6] 加载模型...")
        print(f"  路径: {model_path}")
        
        # 规避 FlashAttention 依赖
        if not use_flash_attn:
            os.environ['USE_FLASH_ATTN'] = '0'
            os.environ['USE_FLASH_ATTENTION'] = '0'
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="eager"
        )
        model.eval()
        print("  ✓ 模型加载成功")
        
        # 2. 加载 tokenizer
        print(f"\n[2/6] 加载 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("  ✓ Tokenizer 加载成功")
        
        # 3. 提取模型配置
        print(f"\n[3/6] 提取模型配置...")
        model_config = extract_model_config(
            model=model,
            model_name=model_name,
            max_seq_length=2048
        )
        
        print(f"  模型名称: {model_config.model_name}")
        print(f"  层数: {model_config.num_layers}")
        print(f"  隐藏层维度: {model_config.hidden_dim}")
        print(f"  专家数量: {model_config.num_experts}")
        print(f"  最大序列长度: {model_config.max_seq_length}")
        print(f"  数据类型: {model_config.dtype}")
        print("  ✓ 模型配置提取成功")
        
        #4. 创建缓冲区
        print(f"\n[4/6] 创建缓冲区...")
        buffer = create_buffer(
            model_config=model_config,
            pattern="attn_gate",
            buffer_size_gb=2.0,
            device="cuda"
        )
        print("  ✓ 缓冲区创建成功")
        
        # 5. 创建采样器
        print(f"\n[5/6] 创建采样器...")
        sampler = OnlineSampler(
            model=model,
            tokenizer=tokenizer,
            dataset_path=dataset_path,
            buffer=buffer,
            pattern="attn_gate",
            batch_size=1,
            max_seq_length=2048,
            trust_remote_code=True
        )
        print("  ✓ 采样器创建成功")
        
        # 6. 启动采样并验证数据
        print(f"\n[6/6] 启动采样并验证数据...")
        print(f"  目标采样数量: {max_samples}")
        print(f"  超时时间: {timeout} 秒")
        
        sampler.start()
        
        import time
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if buffer.get_size() >= max_samples:
                print(f"  ✓ 已采集 {buffer.get_size()} 个样本")
                break
            time.sleep(0.5)
        
        if buffer.get_size() == 0:
            print("  ✗ 未能采集到样本")
            sampler.stop()
            buffer.stop()
            del model
            del tokenizer
            torch.cuda.empty_cache()
            return False
        
        # 验证数据格式
        print(f"\n  验证数据格式...")
        from online_sample.predictor_interface import create_predictor_interface
        
        interface = create_predictor_interface(
            buffer=buffer,
            pattern="attn_gate",
            batch_size=1,
            timeout=None
        )
        
        batch = interface.get_batch()
        
        if batch is None:
            print("  ✗ 无法从缓冲区读取数据")
            sampler.stop()
            buffer.stop()
            del model
            del tokenizer
            torch.cuda.empty_cache()
            return False
        
        data = batch[0]
        
        print(f"  样本数据:")
        print(f"    Tokens shape: {data.tokens.shape}")
        print(f"    Gate logits shape: {data.gate_logits.shape}")
        print(f"    Attn hidden states shape: {data.attn_hidden_states.shape}")
        print(f"    Seq lengths: {data.seq_lengths.shape}")
        
        # 验证数据
        is_valid = data.validate("attn_gate")
        
        if is_valid:
            print("  ✓ 数据格式验证通过")
        else:
            print("  ✗ 数据格式验证失败")
            sampler.stop()
            buffer.stop()
            del model
            del tokenizer
            torch.cuda.empty_cache()
            return False
        
        # 停止采样
        sampler.stop()
        
        # 清理资源
        buffer.stop()
        del model
        del tokenizer
        torch.cuda.empty_cache()
        
        print("\n" + "=" * 80)
        print(f"✓ 模型 {model_name} 测试通过！")
        print("=" * 80)
        return True
        
    except Exception as e:
        print(f"\n✗ 模型 {model_name} 测试失败！")
        print(f"错误信息: {e}")
        import traceback
        traceback.print_exc()
        
        try:
            if 'model' in locals():
                del model
            if 'tokenizer' in locals():
                del tokenizer
            if 'buffer' in locals():
                buffer.stop()
            if 'sampler' in locals():
                sampler.stop()
            torch.cuda.empty_cache()
        except:
            pass
        
        return False


def main():
    """
    主函数：测试Phi-tiny-MoE-instruct模型
    """
    models = [
        {
            "path": "/data1/gx/MoE-predict/models/Phi-tiny-MoE-instruct",
            "name": "Phi-tiny-MoE-instruct",
            "dataset": "/data1/gx/MoE-predict/dataset/processed/test/mmlu.jsonl",
            "timeout": 600,
            "use_flash_attn": False
        }
    ]
    
    results = {}
    
    for i, model_info in enumerate(models, 1):
        print(f"\n\n{'#' * 80}")
        print(f"# 测试进度: {i}/{len(models)}")
        print(f"{'#' * 80}\n")
        
        success = test_model(
            model_path=model_info["path"],
            model_name=model_info["name"],
            dataset_path=model_info["dataset"],
            max_samples=3,
            timeout=model_info.get("timeout", 300),
            use_flash_attn=model_info.get("use_flash_attn", False)
        )
        
        results[model_info["name"]] = success
    
    # 打印总结
    print("\n\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    for model_name, success in results.items():
        status = "✓ 通过" if success else "✗ 失败"
        print(f"  {model_name}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ 所有模型测试通过！")
    else:
        print("✗ 部分模型测试失败，请检查错误信息")
    print("=" * 80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
