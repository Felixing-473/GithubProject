# 交付检查表

交付包包含以下项目，请在交付前确认：

- [ ] 源码目录 `env/`（包含 `flood_env.py`）
- [ ] 训练/评估脚本：`run_smoke_ppo.py`, `train_ppo.py`, `eval.py`  
- [ ] Agents：`agents/ppo.py`  
- [ ] 配置文件：`configs/default.json`, `configs/human_low.json`, `configs/human_high.json`  
- [ ] 测试：`tests/test_env.py`（pytest 全部通过）  
- [ ] 参数搜索输出：`param_search_results.csv`  
- [ ] checkpoints：`checkpoints/ppo_smoke.pt`（示例）  
- [ ] 文档：`README.md`, `ACCEPTANCE.md`, `checklist.md`  

交付验证（建议）
- 运行 `pytest` 并截图/记录通过日志  
- 运行 `run_smoke_ppo.py` 并确认 checkpoint 存在  
- 运行 `param_search.py` 并附上 `param_search_results.csv`  
