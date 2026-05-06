# OR-WM Studio 组会 Demo 说明稿

本文档用于组会演示。建议把它作为讲稿使用，演示时不要把项目说成“已经完成完整越野世界模型”，而是强调：**目前已经搭好了越野世界模型实验工作台的框架闭环，真实模型和真实数据可以逐步替换接入。**

## 1. 项目定位

OR-WM Studio 的目标是做一个“越野世界模型实验工作台”，不是直接重写 BeamNG、CARLA 或完整游戏引擎。

当前 demo 重点展示：

```text
数据导入 -> 数据质量检查 -> 场景浏览/标注 -> 合成场景生成
-> BEV/地形重建 -> 轻量模型训练/预测 -> toy RL 回放
-> Run/Job 记录与可追溯评估
```

可以这样对老师介绍：

> 这个 demo 的核心不是把所有越野模型一次性做完，而是先把数据、模型、任务、结果和界面闭环搭起来。后续更换数据集、替换模型、接入远程 GPU 或真实重建算法时，不需要重写整个平台。

## 2. 当前已经实现的功能

| 模块 | 当前状态 | 说明 |
|---|---|---|
| Web demo 界面 | 已实现 | React + Vite + TypeScript，含 Dashboard / Scene Lab / Dataset / Runs |
| FastAPI 后端 | 已实现 | 提供数据、场景、重建、模型、RL、车辆、run/job 等 API |
| SQLite 记录 | 已实现 | 保存 annotations、vehicles、runs、jobs |
| 内置 demo 数据 | 已实现 | `seq_0001` 可直接跑通界面，不依赖下载大数据 |
| RUGD-style 数据导入 | 已实现 | 支持 RGB + semantic mask 导入，适合 terrain/traversability |
| TartanDrive-style mini 导入 | 已实现 | 支持 CSV/flat mini subset 的 pose/action，适合轨迹预测 |
| Source Card | 已实现 | 每个 sequence 显示来源、license/citation、sensors、限制 |
| Dataset Quality | 已实现 | 检查 images、labels、poses、actions、lidar、depth、calibration 等 |
| Model Catalog | 已实现 | 根据当前 sequence 判断哪些模型 ready / blocked / placeholder / mock |
| 参数编辑 | 已实现 | 每个 launch action 可以展开 `params` 编辑 JSON 参数 |
| Job Registry | 已实现 | 异步 job、状态、进度、日志、取消请求 |
| Run Registry | 已实现 | 记录每次训练/预测/生成结果、指标、artifacts、provenance |
| Run Compare | 已实现 | 对同类 run 的标量指标做简单对比 |
| Scene Lab | 已实现 | 地形/土壤/天气/任务/Prompt 控制，生成合成图和 BEV |
| Mock BEV 重建 | 已实现 | 可生成 occupancy、heightmap、traversability、risk |
| Terrain 轻量模型 | 已实现 | TinyMLP / color prototype 风格的 terrain/traversability baseline |
| Trajectory 轻量模型 | 已实现 | TinyTrajGRU baseline，适合 TartanDrive-style pose/action |
| Toy RL 回放 | 已实现 | 简化 policy replay，用于界面和闭环演示 |
| 车辆配置 | 已实现 | JSON 参数保存和编辑 |

## 3. 目前仍是示意或接口占位的功能

| 功能 | 当前状态 | 为什么先这样做 |
|---|---|---|
| Generate Video | planned，占位按钮 | 本机 AMD 显卡不适合直接跑常见 CUDA diffusion 视频模型；先保留接口位置 |
| Diffusion 视频生成 | 未接真实模型 | 后续建议先接 mock video adapter，再接远程 GPU / 云 API |
| 真实 LiDAR BEV 重建 | 未实现，只是规划接口 | 当前重建是 mock/synthetic，后续接 RELLIS-3D LiDAR + calibration |
| Depth 重建 | 未实现，只是占位 | 后续可接 depth map 或 stereo/depth estimator |
| 完整 TartanDrive 原始格式 | 部分实现 | 目前支持 CSV/flat mini subset，还没解析所有原始发布格式 |
| 完整 annotation 标注器 | 未实现 | 现在是标签摘要保存，还没有像 Label Studio 那样做 mask/polygon 编辑 |
| 车辆动力学真实接入 | 部分实现 | 车辆参数已能保存，但还没深度影响 world model/RL/风险评估 |
| 真正可中断训练 | 部分实现 | Job 支持 cancel request，但重型 adapter 还需要 cooperative checkpoint |
| 强 world model | 未实现 | 当前是 tiny/mock baseline，后续替换成 action-conditioned BEV predictor |

组会时可以强调：

> 这些未实现部分没有在界面中伪装成真实结果。未接入的指标显示 NaN，mock/toy/placeholder 都通过 provenance 标出来。

## 4. 页面介绍

### Dashboard

主页是主闭环视图，重点给老师看“世界模型实验平台”的整体感。

左侧：

- `Data Workspace`：选择当前 sequence。
- `Current frame`：当前播放帧。
- `Source`：数据来源类型。
- `Open Dataset`：进入数据导入与质量页面。
- `Open Scene Lab`：进入合成场景页面。
- `Current Labels`：当前 sequence/frame 的 terrain、soil、weather、task 标签。
- `Save label`：保存当前标签摘要。
- `Dataset Quality`：显示当前数据缺什么、哪些是 placeholder。

中间：

- `front`：前视图。
- `bev`：BEV / occupancy / risk。
- `terrain`：terrain perception 输出，含 semantic / traversability / risk / overlay。
- `trajectory`：轨迹预测图和 ADE/FDE。
- `recon`：重建结果，含 heightmap / traversability / risk。
- `predict`：world model 预测输出。
- `replay`：toy RL policy replay。

右侧：

- `Model Catalog`：当前数据能跑哪些模型，哪些 blocked，为什么 blocked。
- `params`：展开后可编辑模型启动参数。
- `Jobs`：异步任务状态、进度、日志、取消。
- `Runs`：历史运行记录。
- `Compare`：对 run 的指标做对比。
- `任务统计`：成功率、碰撞率、路径长度、风险等，未连接时显示 NaN。
- `训练曲线`：模型训练曲线。
- `事件日志`：RL replay 或任务事件。
- `车辆导入`：编辑车辆 JSON 参数。

### Scene Lab

这是合成数据和未来 diffusion 视频生成的位置。

左侧按钮：

- `Terrain`：荒漠、森林、山地、泥泞、雪地。影响合成场景主题。
- `Soil`：沙土、黏土、砾石、冰面、腐殖土。后续可映射摩擦、打滑、风险。
- `Weather`：晴天、雨天、雾霾。后续可映射图像退化和传感器不确定性。
- `Task`：越野小径、林间穿行、坡道、障碍绕行。影响任务模板和路径分布。
- `Prompt`：自由文本补充场景条件。
- `Generate Image`：生成合成前视图，走异步 job。
- `Generate BEV`：生成 synthetic BEV / occupancy / risk，走异步 job。
- `Generate Video`：目前是 planned，占位。后续接 mock video 或远程 diffusion。
- `Save label`：把当前 scene spec 保存为标签。

注意：如果按钮选的是晴天，但 prompt 写了 rain / fog，会出现 warning。这是为了避免标签和 prompt 冲突。

### Dataset

这是数据导入和质量检查页面。

左侧：

- `sequence select`：切换 sequence。
- `RUGD root path` + `Import RUGD`：导入 RUGD-style RGB + mask。
- `TartanDrive mini root path` + `Import TartanDrive`：导入 pose/action CSV mini subset。

中间：

- `Source Card`：数据来源、许可、引用、传感器、已知限制。
- `Dataset Quality`：检查当前 sequence 是否有 images、labels、poses、actions、lidar、depth、calibration。
- `Sequence Metadata`：显示 metadata.json。

右侧：

- `Dataset-Enabled Models`：根据当前数据质量显示哪些模型 ready / blocked。

### Runs

这是实验追踪页面。

- `Runs`：列出所有运行记录。
- 点击 run：打开 detail drawer，查看 provenance、artifacts、metric summary、raw metrics、config、export bundle。
- `Compare`：按 kind/source 对指标做对比。
- `Jobs`：显示异步任务队列。
- `Refresh jobs/runs`：刷新状态。
- `Cancel`：对 queued/running job 发取消请求。

## 5. 建议组会演示流程

### 演示前启动

后端：

```powershell
npm run dev:backend
```

前端：

```powershell
npm run dev:frontend
```

打开：

```text
http://127.0.0.1:5173
```

### 8-10 分钟演示路线

1. **开场定位**

   说明这是“越野世界模型实验工作台”的 MVP，不是完整游戏引擎。

2. **Dashboard 总览**

   选择 `seq_0001` 或 `rugd_mini`，展示前视图、BEV、Dataset Quality、Model Catalog。

   重点讲：

   - 哪些数据流已经有。
   - 哪些模型 ready。
   - 哪些 blocked，以及 blocked 原因。
   - 未接指标显示 NaN，避免误导。

3. **Dataset 页面**

   切到 `Dataset`。

   展示：

   - Source Card。
   - Dataset Quality。
   - RUGD / TartanDrive-style 导入入口。

   讲：

   > RUGD 更适合语义和可通行性，TartanDrive 更适合轨迹和动力学。

4. **Scene Lab 合成场景**

   切到 `Scene Lab`。

   操作：

   - 选择 `mountain / gravel / sunny / trail`。
   - 修改 prompt。
   - 点击 `Generate Image` 或 `Generate BEV`。
   - 观察 Jobs 面板进度。
   - 完成后看中间图像/BEV。

   讲：

   > 这里是合成数据支路，后续 diffusion 视频生成也放这里，不和真实数据评估混在一起。

5. **Model Catalog 启动模型**

   回到 `Dashboard`。

   展开某个模型的 `params`。

   推荐现场跑较快的：

   - `Reconstruct`
   - `Predict WM`
   - `Run Policy`

   如果要跑训练，把 `epochs` 调小。

   讲：

   > 现在所有模型动作都走 Job Registry，所以后续换成更慢的 diffusion 或 LiDAR 重建，不需要重做 UI。

6. **Runs 页面看结果**

   切到 `Runs`。

   展示：

   - Jobs 状态。
   - Runs 记录。
   - Compare。
   - 点击某个 run 打开 detail drawer。

   重点讲：

   - provenance。
   - metrics。
   - artifacts。
   - config。
   - export bundle。

7. **总结**

   说明当前已经完成平台骨架，后续重点是替换真实模型和真实传感器数据。

## 6. 推荐讲法：当前成果

可以按这段讲：

> 目前已经实现的是一个可运行的越野世界模型实验工作台。它支持内置 demo 数据、RUGD-style 语义数据导入、TartanDrive-style 轨迹动作数据导入、数据质量检查、source card、场景生成、mock BEV 重建、轻量 terrain 模型、轻量轨迹预测模型、toy RL 回放、车辆配置、run 记录和异步 job 管理。
>
> 这个系统的重点是把从数据到模型到评估的闭环搭起来。后续接真实 LiDAR 重建、diffusion 视频生成、action-conditioned world model 或远程 GPU 模型时，可以复用现有的 Dataset / Model Catalog / Job / Run 这套接口。

## 7. 推荐讲法：哪些还是示意

可以按这段讲：

> 当前有几部分还是示意接口。视频生成还没有接真实 diffusion，因为本地是 AMD 显卡，直接跑主流 CUDA 视频模型不现实，所以先把 Generate Video 放在 Scene Lab 作为 planned adapter。真实 LiDAR/Depth 重建还没接，目前 BEV reconstruction 是 mock/synthetic。车辆参数目前能导入和编辑，但还没有深度进入动力学模型。标注也只是标签摘要，还不是完整 mask/polygon 标注器。
>
> 这些功能在界面里不会伪装成真实结果，未接通的指标会显示 NaN，mock/toy/placeholder 会通过 provenance 显示出来。

## 8. 后续优化路线

建议按下面顺序继续：

1. **Video Generation mock adapter**
   - 先不跑真实 diffusion。
   - 用 keyframe 做 deterministic GIF/MP4。
   - 接通 `/api/video-generation/run`。
   - 输出 synthetic video artifact。

2. **远程 diffusion adapter**
   - 本地 AMD 默认不跑。
   - 后续接云 GPU、实验室服务器、AutoDL 或远程 API。
   - backend 可配置为 `mock / remote_api / local_diffusers`。

3. **RELLIS-3D LiDAR BEV**
   - 导入 LiDAR、calibration、pose。
   - 做 point cloud -> heightmap / occupancy / traversability。
   - 把 reconstruction 从 mock 提升到 real_data。

4. **完整 TartanDrive 支持**
   - 支持原始发布格式。
   - 加 IMU、wheel odometry。
   - 增加 constant velocity / bicycle model / TinyTrajGRU 对比。

5. **Action-conditioned world model**
   - 输入 occupancy + action + vehicle state。
   - 输出 future occupancy / ego motion / risk / uncertainty。

6. **标注页面增强**
   - RGB/mask/overlay 叠加查看。
   - 类别 remap。
   - 人工修正和导出 annotation。

7. **更正式评估体系**
   - terrain IoU / F1。
   - trajectory ADE/FDE。
   - BEV IoU。
   - risk calibration。
   - planner success rate。

## 9. 演示时容易被问的问题

### Q1：这是不是 world model？

回答：

> 当前不是一个完整的大型视频 world model，而是一个越野 world model 实验平台。里面已经有轻量 world model baseline 和预测接口，后续可以替换成更强的 action-conditioned BEV dynamics model。

### Q2：为什么不直接接 diffusion 视频模型？

回答：

> 本机是 AMD 显卡，主流视频 diffusion 模型通常依赖 CUDA，直接本地跑不稳定也不现实。所以先把 video generation 放成 adapter 接口，下一步先接 mock video，之后可以接远程 GPU。

### Q3：哪些是真实数据？

回答：

> RUGD-style 和 TartanDrive-style mini 导入是面向真实公开数据格式的 adapter。界面里通过 Source Card 和 provenance 区分 real_data、synthetic、mock、toy_env、placeholder。

### Q4：现在的重建是真实的吗？

回答：

> 当前 BEV 重建主要是 mock/synthetic，用来跑通 UI 和数据结构。真实重建下一步要接 RELLIS-3D 的 LiDAR、pose 和 calibration。

### Q5：Codex 在这里起什么作用？

回答：

> Codex 主要用于快速生成和迭代工程框架：前端工作台、FastAPI 后端、数据格式、adapter、mock baseline、job/run registry 和文档。真正的研究价值在于把这些接口变成可替换的实验平台，后面逐步接真实模型和数据。
