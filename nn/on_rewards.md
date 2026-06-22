Good question. Let me break down the math.

`C_STEP = 0.01` is a **positive** number that gets subtracted: `r -= C_STEP` → `r -= 0.01`, so each step incurs a **-0.01 penalty**. Yes, this nominally encourages fewer pokes.

But does it *actually* matter? Let's compare the forces at play:

| Pressure | Magnitude (30-step episode) | Effect |
|----------|---------------------------|--------|
| $\gamma = 0.95$ discounting | success bonus decays from $10 \to 10 \cdot 0.95^{29} \approx 2.26$ | **Strong** — early success is much more valuable |
| $C_{\text{step}} = 0.01$ penalty | total penalty $\le 30 \cdot 0.01 = 0.30$ | **Negligible** — 3% of the success bonus |

The $\gamma$ discount factor already provides a powerful incentive for shorter episodes: succeeding in 5 steps vs 30 steps changes the discounted success bonus from $10 \cdot 0.95^4 \approx 8.15$ to $\approx 2.26$. That's a \$5.89 difference — massively overshadowing the \$0.30 max step penalty.

So in practice, `C_STEP` is a **tiebreaker** — it breaks symmetry between two policies that achieve success in the same number of steps but one wastes movement. During random exploration (high $\epsilon$), it also gives a slight negative signal so the agent doesn't learn to prefer poking forever when it can't reach the target. But $\gamma$ does the real work.