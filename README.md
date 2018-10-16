[WIP]TouhouPlayer 基于OpenCV的東方红魔乡自动避弹AI
============
一个AI用于自动游玩東方红魔乡(http://en.wikipedia.org/wiki/The_Embodiment_of_Scarlet_Devil).
原作者 feinomenon

Need Module
------------
pip install opencv-python
pip install pywin32
pip install twisted

Usage
------------
先启动游戏, 启动AI(`python player.py`), 然后切换到游戏窗口 AI会自动进入游戏


To-Do
------------
* 读取内存获取自机坐标，残机数，Bomb数
* 通过directX Hook获取更简洁的游戏界面
* 将远古Python2.7代码移植到Python3

Change Log
------------
2018/10/13
将虚拟输入方式改为direct
增加自机范围 （*由于无法获取到准确坐标 判定点产生了误差

2018/10/14
增加OpenCV实时显示判定点
增加OpenCV运动物体追踪实现判断（*由于背景也在运动 部分背景中物体也被识别

2018/10/16
将python2.7代码移植到python3.7