# 工作日志

记录 my-agent 项目搭建过程中的关键决定，及背后的理由。跟 README.md 分开维护——README 说"现在是什么样"，这份文档说"怎么走到这一步、为什么"。

对照目录（只读）：/Users/jj/Downloads/GOAI agent/JKRec
工作目录（可写）：/Users/jj/Desktop/RERand/my-agent

## 2026-09-05

### 决策 1：复现纪律

选择：A —— 全程手打，不复制粘贴 JKRec 里任何文件/代码/配置。
理由：不放心自己能分清"数据"和"逻辑"的界限，用"是不是自己手敲的"这个机械标准代替判断。

### 决策 2：Mac 底子核实

选择：B —— 芯片、python3 版本/来源、Xcode CLT、Homebrew 三项都查清楚。
结果：Apple M5（arm64）；裸命令 python3 不稳定（同一台机器不同会话会解析到 conda 3.14.6 或 Homebrew 3.14.7）；Xcode CLT 与 Homebrew 均已安装；另装了 Homebrew 的 python@3.12（3.12.14）供本项目专用。

### 决策 3：.venv 建立方式

选择：B —— 不改全局 conda 设置，每次干活前手动 conda deactivate，再 activate 项目自己的 .venv。
结果：.venv 用 python3.12 建成功，激活后解释器指向 .venv 内部。

### 决策 4：密钥策略

选择：C —— 现在就建 .gitignore 排除 .venv 和 .env，但不建 .env.example，等真的需要密钥字段时再补。
理由：现阶段判断不需要密钥，但"密钥不能进 git"这道防线值得现在就焊死。

### 决策 5：版本控制启用时机

选择：A —— 现在就 git init，并立刻做第一次提交。
理由：想留下完整的证据链，不想让"什么时候开始留痕"这件事往后拖。

### 决策 6：git 提交身份

选择：B —— 只在本仓库配置身份，不改全局设置。
值：user.name = ygg234，user.email = ljing00234@gmail.com。

### 决策 7：环境验收标准

在原有"能 activate venv、python 能 import、工作路径是 my-agent、对照路径是 JKRec"基础上，加入冒烟测试（.venv 内能装包、能 import）与本文档作为决定留痕的载体。

### 决策 8：冒烟测试结果

在 .venv 中安装 wheel 并成功 import，版本 0.48.0，验证 pip → 安装 → import 整条链路可用。

### 决策 9：pip 缓存权限修复

发现 ~/Library/Caches/pip 权限异常（判断为此前用 sudo 运行 pip 遗留），已用 chown 改回当前用户所有，恢复缓存正常工作。
