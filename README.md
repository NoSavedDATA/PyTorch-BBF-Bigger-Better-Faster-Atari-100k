Unofficial code for "Bigger, Better, Faster: Human-level Atari with human-level efficiency". arXiv: https://arxiv.org/pdf/2305.19452.pdf

Possible replication of the Atari 100k results.

<hr>

Current results:

Alien - 1184.4 IQM - 1184.5 Mean Score (seed 7779)

Assault - 1905.32 IQM - 1918.73 Mean Score (seed 7783)

BankHeist - 509.0 IQM - 603.2 Mean Score (Seed 8012)

BattleZone - 29620.0 IQM - 30610.0 Mean Score (Seed 8234)

Boxing - 97.0 IQM - 96.74 Mean Score (Seed 7800)

Breakout - 299.0  IQM - 292.09 Mean Score (Seed 7803)

Chopper Command - IQM -  Mean Score (Seed 8907)

Experiments were runned over 100 eval seeds for 1 training seed.

<br>

Few modifications were made at the code during these reports, so the results before BattleZone can be inconsistent. I will update results for Alien, Assault and BankHeist later.

<br>

1 environment training takes 7 hours and a few minutes on a 4090.

<hr>

Install torch with version > 2.0

Install my library at: https://github.com/NoSavedDATA/NoSavedDATA

Also:
```
pip install gymnasium[accept-rom-license, atari]
```
