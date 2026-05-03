# OR-WM Studio ML Helpers

This folder keeps the first version intentionally lightweight:

- `scene_generation.py` generates synthetic front-view images, BEV occupancy, heightmap, traversability, and risk maps using only the Python standard library.
- `reconstruction.py` exposes a mock BEV reconstruction entry point with the same artifact shape as a future LiDAR/depth pipeline.
- `world_model.py` provides a deterministic training/prediction fallback and an optional tiny PyTorch model placeholder.
- `rl_env.py` produces a Gym-like deterministic policy replay and training curve when PPO dependencies are unavailable.
- `traversability_model.py` trains a lightweight color-prototype terrain model from RUGD-style semantic masks and predicts semantic groups, traversability, risk, and overlay assets.
- `lightweight_segmentation.py` trains a NumPy TinyMLP terrain classifier from RGB/semantic pairs, producing a real learned lightweight segmentation baseline without requiring PyTorch.
- `rugd_adapter.py` and `import_rugd_dataset.py` import downloaded RUGD-style public data into the OR-WM sequence layout.
- `trajectory_prediction.py` trains and runs a PyTorch TinyTrajGRU baseline for ego trajectory prediction from recent pose history.

For research use, replace the fallback functions with real PyTorch/Gymnasium implementations while keeping the artifact contracts stable.
