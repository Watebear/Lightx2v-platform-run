# MLU590 推理脚本示例

每个案例目录包含 `run_*.sh` 和 `config.json`。Shell 使用 LightX2V 主仓库的推理入口；多卡案例使用 `torchrun`。Hunyuan3D Paint 使用主仓库的 `tools/postprocess/postprocess_paint.py`。

`/data/LightX2V`、`/data/models/...` 和输入素材路径是示例地址。脚本供阅读和迁移参数时参考，未在 MLU590 上按当前入口重新运行。原有 Benchmark 的 Compile/Warmup、两次 Request 和评分记录不由这些单次推理脚本复现，历史测试结果未改动。
