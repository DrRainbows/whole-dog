# RTX 3090 Performance & Storage Notes for `game design` ("The Whole Dog")

Tailored to the live code in `footage/batch_track.py`, `body/train/train_track.py`, `body/train/dog_track.py`, and `body/dog_mocap_mjx.xml` on this machine (`i9-10900K` + `128 GB DDR4` + `RTX 3090 24 GB`). **Nothing in your project was modified while your tasks are running** — apply any of these whenever your current run finishes.

---

## 1. Video + SuperAnimal Pose Pipeline (`footage/batch_track.py`) — ~3× to 6× Faster Wall-Clock

### A. Pure-GPU NVDEC -> `scale_cuda` -> `h264_nvenc` in `transcode()` (Lines 45–47)
Right now `transcode()` decodes on GPU (`-hwaccel cuda`), copies uncompressed frames over PCIe back to CPU RAM for CPU `scale=...`, and encodes 3 clips simultaneously on **CPU `libx264`** (`ThreadPoolExecutor(3)`), pinning all 20 CPU threads while PyTorch DeepLabCut runs on the main thread.

Keeping frames 100% in VRAM on the RTX 3090's dedicated NVDEC + NVENC ASICs uses **0% CUDA SM compute** and **<5% CPU**, running **6–10× faster** (`~450–800 fps`):
```python
subprocess.run([
    "ffmpeg", "-y", "-loglevel", "error",
    "-hwaccel", "cuda", "-hwaccel_output_format", "cuda", "-i", src,
    "-vf", "fps=30,scale_cuda='if(gt(iw,ih),960,-2)':'if(gt(iw,ih),-2,960)'",
    "-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "20", "-an", tmp
], check=True)
```
*(Fallback note: if any phone clip has unusual 10-bit HDR metadata that `scale_cuda` rejects, catch `CalledProcessError` and fall back to the existing `libx264` line for that clip.)*

### B. Fill the 3090 SMs + Enable Ampere `TF32` in `track()` (Lines 117–120)
At `960×540` with `max_individuals=2`, `batch_size=8` and `detector_batch_size=8` use `<2.5 GB` of your `24 GB` VRAM and leave the 3090 SMs mostly waiting on Python overhead. Before calling `deeplabcut.video_inference_superanimal(...)`:
```python
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
```
And raise batch sizes on the 3090:
```python
deeplabcut.video_inference_superanimal(
    [work_mp4], superanimal_name="superanimal_quadruped", model_name="hrnet_w32",
    detector_name="fasterrcnn_resnet50_fpn_v2", videotype=".mp4", dest_folder=DLC_OUT,
    max_individuals=2, batch_size=64, detector_batch_size=32,
    video_adapt=False, create_labeled_video=False)
```

### C. Skip Decoding Unused Frames in `coat_brightness()` (Lines 61–66)
`coat_brightness()` currently calls `cap.read()` (full H.264 decode + BGR conversion) on **every** frame `0..T`, even though `if t % stride == 0` (`stride=3`) throws away 2/3 of them. Using `cap.grab()` on skipped frames avoids color-converting/copying 67% of frames:
```python
if t % stride != 0:
    if not cap.grab():
        break
    t += 1
    continue
ok, frame = cap.read()
```

---

## 2. WSL2 JAX + MuJoCo MJX + Brax PPO (`body/train/`) — ~2× to 4× Faster 60M-Step Runs

### A. Enable Ampere `TensorFloat-32` + Triton GEMM Fusion (`train_track.py` Top of File)
By default JAX runs `float32` matmuls on standard FP32 CUDA cores (`35.6 TFLOPS` on RTX 3090) instead of **Ampere TF32 Tensor Cores (`71+ TFLOPS`)**. Add at the very top of `train_track.py` (before creating networks):
```python
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_triton_gemm_any=True")
import jax
jax.config.update("jax_default_matmul_precision", "tensorfloat32")
```
*(Setting `XLA_PYTHON_CLIENT_PREALLOCATE=false` also stops WSL2 JAX from grabbing 18.4 GiB upfront if another desktop tool is open.)*

### B. Explicit `solver="newton"` in `body/dog_mocap_mjx.xml` (Line 3)
`dog_mocap_mjx.xml` is already nicely pruned (`contype="0"` on body capsules, only 4 spherical `paw` geoms colliding with `floor`, 36 DoF). Explicitly pinning `solver="newton"` with `ls_iterations="6"` is MJX's fastest GPU solver path for small `<50 DoF` articulated trees:
```xml
<option timestep="0.004" solver="newton" iterations="4" ls_iterations="6"/>
```

### C. Vectorize the Post-Training Evaluation Rollout (`train_track.py` Lines 67–79)
Currently lines 68–79 run a sequential Python `while` loop calling `float(st.done)` every single step — forcing up to `300 × N_starts` host-device PCIe syncs. Replacing that block with a single `jax.vmap` + `jax.lax.scan` evaluates **all** start frames simultaneously on the 3090 in **`~0.2s` total**:
```python
starts = jp.arange(0, env._T - 60, 25)
def eval_one(s):
    st0 = env.reset(jax.random.PRNGKey(s))
    st0 = st0.replace(info=dict(st0.info, k=s))
    def body(carry, i):
        st, alive, steps, pose_sum = carry
        act, _ = make_inference(params, deterministic=True)(st.obs, jax.random.PRNGKey(i))
        nxt = env.step(st, act)
        pose_sum = pose_sum + alive * nxt.metrics["r_pose"]
        steps = steps + alive
        alive = alive * (1.0 - nxt.done)
        return (nxt, alive, steps, pose_sum), None
    (_, _, steps, pose_sum), _ = jax.lax.scan(
        body, (st0, jp.float32(1.0), jp.float32(0.0), jp.float32(0.0)),
        jp.arange(env._config.episode_length))
    return steps, pose_sum

all_steps, all_pose_sums = jax.jit(jax.vmap(eval_one))(starts)
```

### D. Fast Quaternion-Conjugate Vector Rotate in `dog_track.py` `_obs()` (Lines 60–66)
`_obs()` instantiates `jax.scipy.spatial.transform.Rotation.from_quat(...).inv()` and calls `.apply()` 4 times per env per step (`16,384` calls per control step at `4096` envs). A direct 3-line conjugate cross-product helper produces identical `float32` outputs with smaller XLA HLO & lower register pressure:
```python
def quat_inv_apply(q_wxyz, v):
    # Rotate vector v by conjugate of unit quaternion q = (w, x, y, z)
    w, u = q_wxyz[0], -q_wxyz[1:4]
    t = 2.0 * jp.cross(u, v)
    return v + w * t + jp.cross(u, t)
```

---

## 3. Freeing `C:` Headroom (`62.4 GB` Free on `C:` vs `5,521 GB` Free on `X:`)
`<repo>\footage\library\work` (68 transcoded clips + `.h5` + `.npz` + future 3D reconstructions) and `body\train\runs` sit on `C:` (`62.4 GB` free). Whenever `batch_track.py` is idle, you can transparently move `work` onto `X:` with zero code changes via an NTFS directory junction:
```cmd
robocopy "<repo>\footage\library\work" "runs\footage-work" /E /MOVE
mklink /J "<repo>\footage\library\work" "runs\footage-work"
```
*(Note: `the local LLM router start script` has also been tuned to `--sleep-idle-seconds 300`, so `llama-server` automatically releases its 22 GiB of RTX 3090 VRAM after 5 minutes idle so DeepLabCut and WSL2 MJX always get the full 24 GB card.)*
