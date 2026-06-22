## How FiLM handles this (correctly)

The features `f(x, y)` vary spatially. The FiLM modulation `γ·f + β` uses the same `γ, β` everywhere, but the **input** to the modulation — `f` — encodes all the spatial information:

```
Pixel (10, 20) = left edge of box  → f encodes "edge, left-facing, near target"
Pixel (100, 50) = right edge of box → f encodes "edge, right-facing, far from target"

Same d̂=(0.8,-0.6) → same γ,β everywhere
BUT: γ·f_left + β ≠ γ·f_right + β   ← different because f differs
     ↓                                    ↓
Q_head sees different patterns      → Q_head can assign different Q-values
```

The spatial discrimination comes from `f`, not from the modulation. `FiLM` amplifies/suppresses **all** features globally by the action, but the Q-head still sees different feature values at different locations.

## What my earlier statement actually got wrong

I said concat is "worse" because the Q-head could learn spurious pixel-dependent action semantics. That's misleading. Both architectures capture spatial variation — concat through explicit param-feature interaction at each pixel, FiLM through global modulation of spatially-varying features. Concat is more parameter-heavy; FiLM bakes in the prior that action semantics are spatially consistent (which is true — `d̂ = (0.8, -0.6)` means the same push direction regardless of where you apply it). The spatial variation in Q-values comes from where you apply it, not from the direction meaning something different at each location.

So your design with FiLM is fine. The spatial discrimination works exactly as intended.