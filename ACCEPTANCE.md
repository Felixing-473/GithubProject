# 验收测试与性能验证清单

目的：定义交付验收标准、自动化测试项与最小可接受性能指标，确保工程交付可运行并具备可复现性。

一、必做自动化测试（必须通过）
- 单元测试：`pytest`（已包含 `tests/test_env.py`），运行命令：

```bash
.venv\Scripts\python.exe -m pytest -q
```

- 验收 smoke-run：运行短时 PPO：

```bash
.venv\Scripts\python.exe run_smoke_ppo.py
```

  - 预期：脚本能在若干分钟内完成并在 `checkpoints/ppo_smoke.pt` 生成文件（无需训练收敛）。

- 参数搜索：

```bash
.venv\Scripts\python.exe param_search.py
```

  - 预期：生成 `param_search_results.csv` 并包含三条场景结果。

二、功能性验收项（手动/自动验证）
- 环境接口：`reset()` 返回长度 29 的观测，`step(action)` 返回 `(obs,reward,done,info)`。
- 奖励函数：在存在重症滞留时，`_compute_reward()` 给出负值（单元测试覆盖）。
- 洪水状态机：step 27-29 触发 WARNING，step 30 触发 FLOODED 并导致自动分流（可通过增加日志或临时运行 `env.render()` 检查）。

三、性能基线（最小可接受）
- 单次 smoke-run（2000 步）能完整运行且生成 checkpoint（无需 GPU）。
- pytest 全部通过。

四、交付物清单（交付时需包含）
- 源码：`env/`, `agents/`, `train_ppo.py`, `run_smoke_ppo.py`, `eval.py`。
- 配置：`configs/`（含三套场景）。
- 测试：`tests/`。
- 运行手册：`README.md`、`ACCEPTANCE.md`、`checklist.md`。

五、验收步骤（建议顺序）
1. 克隆仓库并创建虚拟环境。安装依赖。  
2. 运行 `pytest`，确认全部通过。  
3. 运行 `param_search.py`，确认 `param_search_results.csv` 生成。  
4. 运行 `run_smoke_ppo.py`，确认 `checkpoints/ppo_smoke.pt` 存在。  
