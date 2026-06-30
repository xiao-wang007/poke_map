**Script Editor Workflow**

1. Open Isaac Sim and run this first:
```python
/home/xiao/0_codes/poke_map/env/scene_setup_articulated_vectorized.py
```

This creates `/World/envs/env_0...`, clones the finger/object setup, randomizes object/finger properties if enabled, and settles the scene. Wait until you see the scene settle message before starting training.

2. For **Stage A**, keep [nn/train.py](/home/xiao/0_codes/poke_map/nn/train.py:100) like this:
```python
YAW_TARGET_MODE = "preserve"
C_YAW = 0.1
YAW_REWARD_ENABLED = True
YAW_SUCCESS_ENABLED = False
```

This means: train translation, render the target with preserved yaw, softly penalize yaw drift, but success/done is translation-only.

3. Run [nn/train.py](/home/xiao/0_codes/poke_map/nn/train.py) in Script Editor.

Then start training:
```python
trainer_task = main(num_episodes=2000)
```

Checkpoints save under:
```text
/home/xiao/0_codes/poke_map/checkpoints/
```

4. When Stage A looks good, switch [nn/train.py](/home/xiao/0_codes/poke_map/nn/train.py:100) to **Stage B**:
```python
YAW_TARGET_MODE = "curriculum"
C_YAW = 0.5
YAW_REWARD_ENABLED = True
YAW_SUCCESS_ENABLED = True
```

Use `"curriculum"` if you want orientation difficulty to ramp. Use `"fixed"` if you want full orientation targets immediately.

5. Reload/run [nn/train.py](/home/xiao/0_codes/poke_map/nn/train.py) again in Script Editor, then resume from Stage A weights while clearing replay:
```python
trainer_task = main(
    num_episodes=5000,
    resume="/home/xiao/0_codes/poke_map/checkpoints/ep002000.pt",
    resume_mode="weights",
)
```

That keeps the learned network from Stage A, clears old translation-stage replay, resets exploration schedule, and starts collecting fresh orientation-stage transitions.

For normal crash/interrupt resume within the same stage, use full resume:
```python
trainer_task = main(
    num_episodes=5000,
    resume="/home/xiao/0_codes/poke_map/checkpoints/ep002000.pt",
)
```

One practical note: since `train.py` imports `randomize_object_poses` from the scene setup file, if you edit `scene_setup_articulated_vectorized.py`, rerun the scene setup before starting a new training run.


----------------------

Yes, mostly.

Your implementation can continue with different rewards, but the important detail is the replay buffer.

If you change `c_step`, `c_target_separation`, `c_success`, etc., you should continue with **weights-only resume**, not full resume, because full resume keeps old replay transitions with old rewards already stored.

Use this pattern:

1. Save checkpoint from current stage.
2. Edit `config.yaml`, for example:
   ```yaml
   c_step: 0.04
   max_steps_per_episode: 15
   ```
3. Rerun/reload the training script so `rl/rewards.py` reloads the new config.
4. Resume with `resume_mode="weights"`.

That keeps the learned network weights but clears replay, so the critic starts collecting transitions under the new reward definition.

If you use full resume, it will technically run, but the buffer will contain a mixture of old-reward and new-reward transitions. For curriculum reward changes, that is messy and I would avoid it.

One caveat: your current weights-only resume resets epsilon/noise to `EPS_START` / `SIGMA_START`. That may be okay for a new curriculum stage, but if you want a smoother continuation, we could patch a “weights resume but preserve/set epsilon” option.