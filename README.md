# Snooker
A brief description about snooker environment.

| Action Spacce | Tuple(Discrete(360), Discrete(90)) |
| Observation Space | Tuple(Continuous(22,2)([133, 966], [83, 490]))

# Description
The detail of environment.

# Action Space
- angle: [0, 359] seperate space step 1

- force: [20, 110] seperate space step 1

# Observation Space
The observation contans 22 balls positions with the format (x,y).
- x: [133, 966]
- y: [83, 490]

# Reward Space
- 

# Reference
- https://github.com/skeleta/Snooker