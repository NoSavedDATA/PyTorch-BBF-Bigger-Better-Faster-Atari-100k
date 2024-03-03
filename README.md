Unofficial code for "Bigger, Better, Faster: Human-level Atari with human-level efficiency". arXiv: https://arxiv.org/pdf/2305.19452.pdf

Possible replication of the Atari 100k results.

<hr>

Current results:

Alien - 1184.4 IQM - 1184.5 Mean Score (seed 7779) n-1

Assault - 1905.32 IQM - 1918.73 Mean Score (seed 7783) n-1

BankHeist - 509.0 IQM - 603.2 Mean Score (Seed 8012) n-1

BattleZone - 29620.0 IQM - 30610.0 Mean Score (Seed 8234) n

Boxing - 97.0 IQM - 96.74 Mean Score (Seed 7800) n

Breakout - 299.0  IQM - 292.09 Mean Score (Seed 7803) n

Chopper Command - 5120.0 IQM - 4876.0 Mean Score (Seed 8907) n

Crazy Climber - 16100.0 IQM - 16955.0 Mean Score (Seed 8908) n-1

Demon Attack - 13387.2 IQM - 14651.75 Mean Score (Seed 7811) n-1

Freeway - 24.0 IQM - 24.03 Mean Score (Seed 8259) n-1

Frostbite - 253.7 IQM - 257.4 Mean Score (Seed 8262) n-1

Gopher - 200.0 IQM - 250.0 Mean Score (Seed 8264) n-1

Hero - 3020.0 IQM - 3040.0 Mean Score (Seed 8048) n-1

Jamesbond - 299.0 IQM - 427.5 Mean Score (Seed 8052) n-1

Kangaroo -  IQM -  Mean Score (Seed 8712) n-1





Experiments were runned over 100 eval seeds for 1 training seed.


<br>

1 environment training takes 6 hours and a few minutes on a 4090.

<hr>

Install torch with version > 2.0

Install my library at: https://github.com/NoSavedDATA/NoSavedDATA

Also:
```
pip install gymnasium[accept-rom-license, atari]
```
