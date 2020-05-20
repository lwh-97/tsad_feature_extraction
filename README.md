## TSAD feature extraction

#### tsad_flask代码框架：

* requirements.txt 需要安装的库
  
  ```bash
  pip install -r requirements.txt
  ```

  除了requirements.txt中提到的，还需要安装周期性判断所需的依赖库——[interp-acf](https://github.com/bmorris3/interp-acf)
  
  （interp-acf的Github主页有安装教程）
  
  建议创建一个全新的python3.6环境，以避免不必要的依赖库版本冲突，推荐使用anaconda([清华镜像](https://mirror.tuna.tsinghua.edu.cn/help/anaconda/))

* main/tsad_main.py 运行启动入口，可先运行一下确保环境搭建成功，没有其他bug

* Base.py 是主要的类，其中的_get_feature函数便是在提取特征


