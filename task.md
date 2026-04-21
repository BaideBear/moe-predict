# 工作任务列表

代码任务

- [x] 实现eval脚本，结构和train_predict.py类似，把训练模块换成推理模块即可
- [] 实现不同的predicor结构，并完成注册（LSTM、 Transformer）
- [x] 实现不同的loss函数，并完成loss函数的可替换逻辑（如带top-k加权的loss）
- [] 完成pre-moe的训练代码（逻辑上只需更改online sample的参数就可以了）
- [] 非训练任务的推理逻辑实现

训练任务

- [x] 对mlp结构的不含dropout版本进行训练
- [] 对于新predicor结构进行训练（进行中）
- [] 对loss函数进行消融实验(进行中)
- [] 对pre-moe方式进行训练
- [] predictor超参探索（可选）
