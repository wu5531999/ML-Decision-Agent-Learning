📋 前置要求

•操作系统：Windows 10/11、macOS 12+、Ubuntu 20.04+
•磁盘空间：至少 8GB（含 Python 环境和依赖库）
•网络：稳定的互联网连接（用于下载依赖）
•强制Python版本：3.13.x（已验证3.13.7）


🔧 第一步：安装基础软件

1.1 安装 VS Code 代码编辑器
    1.官方下载地址：https://code.visualstudio.com/
    2.安装时使用默认选项，建议勾选以下两项：
    ☑添加到 PATH（重启后生效）
    ☑右键菜单添加"通过 Code 打开"

1.2 确认 Python 3.13.7 已正确安装
    1.打开终端（Windows：Win+R 输入 cmd；Mac：Cmd+空格 搜索"终端"）
    2.执行命令：python --version
    3.终端显示 Python 3.13.7 即为安装成功
    4.未安装可通过官方地址下载：https://www.python.org/downloads/release/python-3137/
    安装时务必勾选最下方的 "Add Python 3.13 to PATH"


📦 第二步：安装 VS Code 必装插件
打开 VS Code，点击左侧边栏「扩展」图标（四个方块），搜索并安装以下插件：

插件名称	作者	核心作用
Python	Microsoft	Python 3.13 语言支持、代码运行与调试
Pylance	Microsoft	智能代码补全、语法错误检查
Code Spell Checker	Street Side Software	代码拼写检查，避免变量名错误
所有插件安装完成后，重启 VS Code 使插件生效。


🐍 第三步：配置项目专属虚拟环境
虚拟环境可以隔离不同项目的依赖，彻底解决"在我电脑上能跑，在你电脑上跑不了"的问题。

3.1 打开项目文件夹
    1.点击 VS Code 左上角「文件」→「打开文件夹」
    2.选择 ML-Decision-Agent-Learning 根文件夹

3.2 打开 VS Code 内置终端
    •快捷键：Ctrl+（Windows）/ Cmd+（Mac）
    •确认终端路径显示为项目根目录，例如：
        ◦Windows：C:\Users\用户名\ML-Decision-Agent-Learning
        ◦Mac：/Users/用户名/ML-Decision-Agent-Learning

3.3 创建虚拟环境
    在终端中执行以下命令：
        python -m venv venv
    执行完成后，项目根目录会自动生成一个 venv 文件夹。

3.4 激活虚拟环境

# Windows PowerShell
.\venv\Scripts\Activate.ps1
# Windows CMD
.\venv\Scripts\activate.bat
# Mac/Linux
source venv/bin/activate
激活成功标志：终端开头会显示 (venv) 字样。

3.5 选择 VS Code Python 解释器
    1.查看 VS Code 右下角状态栏，找到「Python 3.13.7」字样并点击
    2.在弹出的列表中，选择带有 ./venv 路径的解释器：
        ◦Windows：./venv/Scripts/python.exe
        ◦Mac/Linux：./venv/bin/python
    3.等待 VS Code 加载完成，右下角会显示「已选择解释器：venv」


📥 第四步：一键安装所有依赖
确保虚拟环境已激活（终端开头有 (venv)）

然后在终端执行：
# 升级 pip 到最新版本（解决 3.13 下的安装问题）
python -m pip install --upgrade pip
# 一键安装所有依赖
pip install -r 00_环境配置/requirements.txt
等待安装完成（约 3-5 分钟，取决于网络速度）。
验证依赖安装：执行 pip list，查看列表中是否包含上述所有库及对应版本。


✅ 第五步：验证环境是否配置成功
运行项目中自带的线性回归 Demo，验证环境是否正常：
    1.在 VS Code 中打开文件：01_机器学习基础/监督学习/linear_regression.py
    2.点击右上角的「运行」按钮（三角形图标），或按快捷键 F5
    3.正常结果：
        ◦终端输出：拟合参数：斜率w=2.03, 截距b=4.97 和均方误差
        ◦弹出一个窗口，显示散点图和红色拟合直线
能正常运行并出图，说明环境配置 100% 成功。

❌ 常见问题排查
问题1：提示"找不到满足要求的版本 xxx"
•原因：使用了旧版本的 requirements.txt
•解决：使用本文档提供的 Python 3.13 专属依赖清单
问题2：PowerShell 提示"无法加载文件，因为在此系统上禁止运行脚本"
•解决：以管理员身份打开 PowerShell，执行以下命令，输入 Y 确认，重启 VS Code

Set-ExecutionPolicy RemoteSigned
问题3：matplotlib 画图不显示或报错
•解决：在所有代码文件的最开头添加以下两行：

import matplotlib
matplotlib.use('TkAgg')
问题4：PyTorch 安装失败
•解决：使用官方镜像源单独安装：

pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
📝 补充说明
添加新依赖
如需安装额外的库，在激活虚拟环境后执行：

pip install 库名
pip freeze > 00_环境配置/requirements.txt
这会自动更新依赖清单，确保其他人可以一键复现环境。
更新环境
如果拉取了最新代码，依赖有更新，执行以下命令即可：

pip install -r 00_环境配置/requirements.txt --upgrade