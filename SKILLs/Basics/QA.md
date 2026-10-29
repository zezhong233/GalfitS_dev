# **Frequently asked question & anwser**

#### Q1: 显存满了： "RESOURCE_EXHAUSTED: Out of memory while trying to allocate xxx bytes"
-  A1: 先`nvidia-smi`查看目前显存占用情况（默认使用0卡）。如果有未占满的卡，在`galfits`前面加上`CUDA_VISIBLE_DEVICES={NUM}`来选择对应显卡。`NUM`对应卡号，注意空格。 
```
CUDA_VISIBLE_DEVICES=1 galfits objxx.lyric --workplace wp --fit_method optimizer
```

#### Q2: 