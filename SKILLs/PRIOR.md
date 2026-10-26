# GalfitS prior usage format


## 0. basic grammar
```
galfits objxxx.lyric --workplace results --priorpath {prior_path}
```

## 1. differnt prior type

### 1.1 Gaussian prior 
GalfitS allows you to set an independent gaussian prior to each parameter. 

- Usage: in `xxx.prior` file, starting a line with prefix `GP)` then using a list to contain all the parameter names you want to apply Gaussian prior.

```
GP) ['logM_disk', 'disk_dust2', 'disk_dust_index']
```

- Value passing: Gaussian distribution ~ G($\mu$, $\sigma$) needs two parameters: median and sigma. In galfits frame work, these two parameters are passed in `xxx.lyric` file.
    - in the standard 5-array list, [init, min, max, step, vary], init $\Rightarrow$ mean, step $\Rightarrow$ sigma. 
    ```
    [mean, min, max, sigma, vary]
    ```


**NOTE**: when using gaussian prior, please add `--baysian`, otherwise it will ignore the prior. 


### 1.2 Cosmic SFH prior 
We follow prospector-beta method to implement cosmic sfh prior in **Non-parametric SFH** fitting. It favors a trend that *massive galaxy forms early, while low-mass galaxy forms later*.  


