# PSMFN
# 本模型基于basicsr框架训练
# 模型测试
```python  
python /root/BasicSR/basicsr/test.py -opt /root/BasicSR/options/test/benchmark_PSMFN_x4.yml 
```
# 模型训练
```python  
CUDA_VISIBLE_DEVICES=0 python /root/BasicSR/basicsr/train.py -opt /root/BasicSR/options/train/trainPSMFN_x4.yml
```
