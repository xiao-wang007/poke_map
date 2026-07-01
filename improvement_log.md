Yes, I remember. It has gone **very far** from the first running logs.

**Big Picture**
At the beginning, the problem was not just “RL is hard.” A lot of the training loop was still fighting execution details:

| Stage | What We Saw | Meaning |
|---|---:|---|
| Very first logs | `len=30` every episode, `r/active≈0.15-0.23`, `dt≈240-300s` with 64 envs | Episodes were dragging to max pokes; last envs slowed everything |
| After execution fixes | `len≈14-17`, `done≈123/128`, `active=128→~6`, `r/active≈0.40+` | Termination and poke execution became much healthier |
| Bad policy-takeover run | policy `>90%`, `done≈20`, `active_end≈110`, `dir_po≈65%` | The learned policy collapsed when the heuristic disappeared |
| Current direction-loss run | ep 151-198: `done≈122`, `active_end≈8`, `policy≈28%`, `dir_po≈17%` | Much more stable; direction loss is helping |

**Issues We Found**
The main issues were:

1. Fixed strike distance caused back-and-forth pushing near targets.
2. Episodes waited too long for the last few envs.
3. Contact velocity could become too slow/sustained before we made pokes more impulsive.
4. The prior/heuristic was carrying training, but the policy was not learning good directions.
5. The policy could choose directions pointing away from the target.
6. Half-plane projection alone corrected execution, but did not teach the raw param-head enough.
7. When epsilon decayed too fast, policy takeover caused collapse.

**Improvements Added**
The useful fixes were:

1. Adaptive strike length.
2. `min_active_envs` early stop.
3. Shorter max episode length.
4. Mid-range heuristic velocities.
5. More impulsive strike timing.
6. Target-separation progress reward.
7. Refactor into `rl/rewards.py`, `rl/termination.py`, `rl/replay_buffer.py`, `rl/prior.py`.
8. Policy/prior action line colors.
9. `r_pol`, `r_heur`, `dir_po`, velocity stats.
10. Target-facing half-plane projection.
11. Store `target_dir` in replay.
12. Direction loss on raw actor direction.
13. Slower epsilon decay.

**Did It Work?**
Yes, partly and meaningfully.

The execution/training loop fixes definitely worked. You went from slow, max-length episodes to much cleaner training episodes with high `done` counts and much better `r/active`.

The half-plane projection alone did **not** fully work. It prevented bad execution directions, but the policy still collapsed later.

The direction loss looks like the first fix that actually attacks the learned policy problem. In the current run, `dir_po` is much lower and `done` is still strong while policy usage rises.

But it is not fully solved yet. The policy is still weaker than the heuristic:

```text
r_pol < r_heur
```

and raw `greedy_v` is still low compared with executed `sel_v`, so velocity learning is not perfect either.

My current verdict: **the system is now genuinely learning better than before, but the policy still has to survive policy% > 50% before we call it successful.**