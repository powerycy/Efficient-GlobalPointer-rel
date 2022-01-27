## 引言
1. 苏神Efficient-GlobalPointer原文链接https://spaces.ac.cn/archives/8877
2. 关于torch版本的Efficient-GlobalPointer复现。
## 如何运行
1. Pytorch版本Efficient-GlobalPointer模型的关系抽取版本,之前本人在讯飞大赛的医疗关系抽取大赛中进行过实验。
2. config.ini文件可配置所需的参数
3. globalpointer_train运行，inference可以预测出结果，可进行ddp形式.
4. 解码是使用tplinker的思路，这里感谢Tplinker的作者= =
5. 期待苏神的版本本人实现比较粗暴，等待苏神开源后会自行修改苏神的版本会比较优雅~
