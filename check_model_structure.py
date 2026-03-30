import torch
from transformers import AutoModelForCausalLM

model_path = "/data1/gx/MoE-predict/models/Mixtral-8x7B-v0.1"

print("Loading model to check structure...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

print("\n=== Model Structure ===")
print(f"Model type: {type(model)}")
print(f"Model class: {model.__class__.__name__}")

if hasattr(model, 'model'):
    print(f"\nModel has 'model' attribute")
    if hasattr(model.model, 'layers'):
        print(f"Number of layers: {len(model.model.layers)}")
        
        print("\n=== Checking first few layers ===")
        for i in range(min(3, len(model.model.layers))):
            layer = model.model.layers[i]
            print(f"\nLayer {i}:")
            print(f"  Type: {type(layer)}")
            print(f"  Attributes: {dir(layer)}")
            
            if hasattr(layer, 'mlp'):
                print(f"  Has MLP: {type(layer.mlp)}")
                print(f"  MLP attributes: {[attr for attr in dir(layer.mlp) if not attr.startswith('_')]}")
                
                if hasattr(layer.mlp, 'gate'):
                    print(f"  Has gate: {type(layer.mlp.gate)}")
                else:
                    print(f"  No gate attribute")
                    
                if hasattr(layer.mlp, 'block_sparse_moe'):
                    print(f"  Has block_sparse_moe: {type(layer.mlp.block_sparse_moe)}")
                    if hasattr(layer.mlp.block_sparse_moe, 'gate'):
                        print(f"    block_sparse_moe has gate: {type(layer.mlp.block_sparse_moe.gate)}")
                    print(f"    block_sparse_moe attributes: {[attr for attr in dir(layer.mlp.block_sparse_moe) if not attr.startswith('_')]}")
