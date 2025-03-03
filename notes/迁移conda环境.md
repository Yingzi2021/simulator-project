# 迁移conda环境

（0）准备：

将本机的ssh公钥复制到云主机的`.ssh/authorized_keys`中，这样可以使用在vscode上使用ssh远程连接到云主机

```
ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQDBP1BDQv19ufabgII/jsips4yyclRdcGEZUp61+T6g+XLg+QEO9nxroXpqGNp313QQy3CrOFpZq4DCWHkluOIfxabtCfEFDSc5L2LI+SC3FY5zFFDOzGq/eUL5Fq3g+/sJ03D5x2kMLFD4KD8MR/WQAX4209iFm2CvqpHqI5zN5Km/rwRXPn1cl+ddpAgj+Rw3bdhR+3tgHzoZmbtrWFOcqyxCB5AK50USoXDZMIdSenwLX5Fo198cpmKiW8tsjI214HvDqIpmHxTMBdTOL38StRP+hcXAdoLnCNBk7dK4fRvsSsAxwQGcU09MwEOtitNaZanQrKwtqZfz2ArlooleUftlqOAqchwAu4QQAnhH6dCxkoj4CIq3tG61GI6vfnyYfLRVh+47dhyldtJn44zJq5EPjlSdS2E7RKGRc/vtwmvBp0OIRt6h/JaslYZcP3/CFyne3G8OqinkY+RrK0UAQxp7FNh/O604ObPzL/PPl/xrJK/fmLim7xR3w/1dfp8= 2292753010@qq.com
```

（1）查看目标机器cuda版本并配置环境变量

```
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
```

（2）从ytyang机器导出原环境

```
conda activate myenv
conda env export > environment.yaml
```

（3）在新机器上，使用以下命令来创建与导出文件中相同的环境：

```
conda env create -f environment.yaml
```

这个命令会根据 `environment.yaml` 文件来重新安装所有依赖包，创建一个相同的环境。

（4）激活新环境

```
conda activate myenv
```

（5）安装nsight system

在服务器上下载Linux CLI软件包

```
wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2024_5/NsightSystems-linux-cli-public-2024.5.1.113-3461954.deb
```

安装

```
sudo dpkg -i NsightSystems-linux-cli-public-2024.5.1.113-3461954.deb
```

查看版本

```
nsys --version
```
