**Additional Rewards:** 
* kills up to 8 per floor except for the overworld += 1
* for every iron += 0.5
* for every diamond += 1
* for every coal += 0.3
* for every wood += 0.1
* for every stone with a per episode high mark += 0.01
* for every additional armour piece += 3
* for every additional enchantment of a armour piece += 2
* for every level up stat += 1.5
* for every sapphire/ruby += 1
* for enchanting the bow += 2
* kills scale extra for every floor to encourage further fighting +25%
* bonus for every floor cleared += 3

Depth based Potential based Reward Shaping was used to punish the agent for retreating, to encourage preparing himself in one go before descending, and to discourage retreating into upper floors to avoid death.

$$\text{depth_bonus} = \text{depth_gamma} \cdot \phi(s') - \phi(s)$$

where the potential function $\phi(s)$ is defined as:

$$\phi(s) = w_{\text{depth}} \cdot \text{current_floor}(s)$$

with $w_{\text{depth}} = 2.0$ and $\text{depth_gamma} = 0.999$, meaning descending to a deeper floor yields a positive shaping reward of $\approx +2$, while ascending yields a penalty of $\approx -2$.