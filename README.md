# 洪涝疏散仿真环境（样例实现）

最小可运行实现，包含阶段1（离线仿真）、阶段2（Gym 封装）与阶段3（轻量 PPO 验证性训练）。

快速开始
1. 创建虚拟环境并安装依赖：

```bash
python -m venv .venv
.venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

2. 运行冒烟测试（pytest）：

```bash
.venv\Scripts\python.exe -m pytest -q
```

3. 运行随机策略演示：

```bash
.venv\Scripts\python.exe train.py
```

4. 运行 PPO 短跑（smoke-run，保存 checkpoint）：

```bash
.venv\Scripts\python.exe run_smoke_ppo.py
```

5. 运行参数搜索并生成 CSV 报表：

```bash
.venv\Scripts\python.exe param_search.py
```

6. 评估训练或基线策略：

```bash
.venv\Scripts\python.exe eval.py
```

交付物说明
- `env/flood_env.py`：环境核心实现（reset/step/render）
- `agents/ppo.py`：轻量化 PPO trainer（PyTorch）
- `train.py`：随机策略示例
- `run_smoke_ppo.py`：PPO 短跑入口（2000 steps 默认）
- `train_ppo.py`：PPO 训练脚本（示例）
- `eval.py`：加载并评估模型
- `param_search.py`：轻量参数搜索并输出 `param_search_results.csv`
- `configs/`：三套示例配置（default / human_low / human_high）
- `tests/`：pytest 单元测试（2 个快速测试）
- `notebooks/colab_train.ipynb`：Colab 模板（挂载 Drive、示例命令）

验收与联调
请参阅 `ACCEPTANCE.md` 和 `checklist.md` 获取验收准则与交付检查表。
