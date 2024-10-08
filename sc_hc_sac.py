nohup: ignoring input
Using cuda device
SACPolicy(
  (actor): Actor(
    (features_extractor): FlattenExtractor(
      (flatten): Flatten(start_dim=1, end_dim=-1)
    )
    (latent_pi): Sequential(
      (0): Linear(in_features=17, out_features=256, bias=True)
      (1): ReLU()
      (2): Linear(in_features=256, out_features=256, bias=True)
      (3): ReLU()
    )
    (mu): Linear(in_features=256, out_features=6, bias=True)
    (log_std): Linear(in_features=256, out_features=6, bias=True)
  )
  (critic): ContinuousCritic(
    (features_extractor): FlattenExtractor(
      (flatten): Flatten(start_dim=1, end_dim=-1)
    )
    (qf0): Sequential(
      (0): Linear(in_features=23, out_features=256, bias=True)
      (1): ReLU()
      (2): Linear(in_features=256, out_features=256, bias=True)
      (3): ReLU()
      (4): Linear(in_features=256, out_features=1, bias=True)
    )
  )
  (critic_target): ContinuousCritic(
    (features_extractor): FlattenExtractor(
      (flatten): Flatten(start_dim=1, end_dim=-1)
    )
    (qf0): Sequential(
      (0): Linear(in_features=23, out_features=256, bias=True)
      (1): ReLU()
      (2): Linear(in_features=256, out_features=256, bias=True)
      (3): ReLU()
      (4): Linear(in_features=256, out_features=1, bias=True)
    )
  )
)
---------------------------------
| time/              |          |
|    episodes        | 4        |
|    fps             | 70       |
|    time_elapsed    | 56       |
|    total_timesteps | 4000     |
| train/             |          |
|    actor_loss      | -40.8    |
|    critic_loss     | 1.05     |
|    ent_coef        | 0.317    |
|    ent_coef_loss   | -9.88    |
|    learning_rate   | 0.0003   |
|    n_updates       | 3899     |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 8        |
|    fps             | 67       |
|    time_elapsed    | 118      |
|    total_timesteps | 8000     |
| train/             |          |
|    actor_loss      | -46.4    |
|    critic_loss     | 1.17     |
|    ent_coef        | 0.105    |
|    ent_coef_loss   | -12.2    |
|    learning_rate   | 0.0003   |
|    n_updates       | 7899     |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 12       |
|    fps             | 64       |
|    time_elapsed    | 185      |
|    total_timesteps | 12000    |
| train/             |          |
|    actor_loss      | -45      |
|    critic_loss     | 1.07     |
|    ent_coef        | 0.039    |
|    ent_coef_loss   | -8.68    |
|    learning_rate   | 0.0003   |
|    n_updates       | 11899    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 16       |
|    fps             | 63       |
|    time_elapsed    | 250      |
|    total_timesteps | 16000    |
| train/             |          |
|    actor_loss      | -41.8    |
|    critic_loss     | 0.963    |
|    ent_coef        | 0.0197   |
|    ent_coef_loss   | -1.45    |
|    learning_rate   | 0.0003   |
|    n_updates       | 15899    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 20       |
|    fps             | 62       |
|    time_elapsed    | 320      |
|    total_timesteps | 20000    |
| train/             |          |
|    actor_loss      | -36.5    |
|    critic_loss     | 0.829    |
|    ent_coef        | 0.0146   |
|    ent_coef_loss   | 0.604    |
|    learning_rate   | 0.0003   |
|    n_updates       | 19899    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 24       |
|    fps             | 62       |
|    time_elapsed    | 383      |
|    total_timesteps | 24000    |
| train/             |          |
|    actor_loss      | -31.8    |
|    critic_loss     | 0.737    |
|    ent_coef        | 0.0108   |
|    ent_coef_loss   | 1.06     |
|    learning_rate   | 0.0003   |
|    n_updates       | 23899    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 28       |
|    fps             | 62       |
|    time_elapsed    | 447      |
|    total_timesteps | 28000    |
| train/             |          |
|    actor_loss      | -30      |
|    critic_loss     | 0.585    |
|    ent_coef        | 0.00939  |
|    ent_coef_loss   | -1.18    |
|    learning_rate   | 0.0003   |
|    n_updates       | 27899    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 32       |
|    fps             | 62       |
|    time_elapsed    | 515      |
|    total_timesteps | 32000    |
| train/             |          |
|    actor_loss      | -28.5    |
|    critic_loss     | 0.654    |
|    ent_coef        | 0.0082   |
|    ent_coef_loss   | 1.25     |
|    learning_rate   | 0.0003   |
|    n_updates       | 31899    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 36       |
|    fps             | 61       |
|    time_elapsed    | 584      |
|    total_timesteps | 36000    |
| train/             |          |
|    actor_loss      | -27      |
|    critic_loss     | 0.739    |
|    ent_coef        | 0.00762  |
|    ent_coef_loss   | -0.714   |
|    learning_rate   | 0.0003   |
|    n_updates       | 35899    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 40       |
|    fps             | 61       |
|    time_elapsed    | 649      |
|    total_timesteps | 40000    |
| train/             |          |
|    actor_loss      | -27.8    |
|    critic_loss     | 0.81     |
|    ent_coef        | 0.00782  |
|    ent_coef_loss   | 2.08     |
|    learning_rate   | 0.0003   |
|    n_updates       | 39899    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 44       |
|    fps             | 61       |
|    time_elapsed    | 715      |
|    total_timesteps | 44000    |
| train/             |          |
|    actor_loss      | -27      |
|    critic_loss     | 0.911    |
|    ent_coef        | 0.00974  |
|    ent_coef_loss   | 0.737    |
|    learning_rate   | 0.0003   |
|    n_updates       | 43899    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 48       |
|    fps             | 58       |
|    time_elapsed    | 817      |
|    total_timesteps | 48000    |
| train/             |          |
|    actor_loss      | -31.2    |
|    critic_loss     | 1.27     |
|    ent_coef        | 0.0153   |
|    ent_coef_loss   | 2.24     |
|    learning_rate   | 0.0003   |
|    n_updates       | 47899    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 52       |
|    fps             | 56       |
|    time_elapsed    | 920      |
|    total_timesteps | 52000    |
| train/             |          |
|    actor_loss      | -41.9    |
|    critic_loss     | 2.92     |
|    ent_coef        | 0.0244   |
|    ent_coef_loss   | -0.546   |
|    learning_rate   | 0.0003   |
|    n_updates       | 51899    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 56       |
|    fps             | 54       |
|    time_elapsed    | 1026     |
|    total_timesteps | 56000    |
| train/             |          |
|    actor_loss      | -55.2    |
|    critic_loss     | 2.61     |
|    ent_coef        | 0.0328   |
|    ent_coef_loss   | 0.526    |
|    learning_rate   | 0.0003   |
|    n_updates       | 55899    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 60       |
|    fps             | 52       |
|    time_elapsed    | 1132     |
|    total_timesteps | 60000    |
| train/             |          |
|    actor_loss      | -71.5    |
|    critic_loss     | 2.48     |
|    ent_coef        | 0.038    |
|    ent_coef_loss   | 0.581    |
|    learning_rate   | 0.0003   |
|    n_updates       | 59899    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 64       |
|    fps             | 51       |
|    time_elapsed    | 1241     |
|    total_timesteps | 64000    |
| train/             |          |
|    actor_loss      | -84.8    |
|    critic_loss     | 3.24     |
|    ent_coef        | 0.0424   |
|    ent_coef_loss   | 0.825    |
|    learning_rate   | 0.0003   |
|    n_updates       | 63899    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 68       |
|    fps             | 50       |
|    time_elapsed    | 1350     |
|    total_timesteps | 68000    |
| train/             |          |
|    actor_loss      | -93.3    |
|    critic_loss     | 2.96     |
|    ent_coef        | 0.0472   |
|    ent_coef_loss   | -1.4     |
|    learning_rate   | 0.0003   |
|    n_updates       | 67899    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 72       |
|    fps             | 49       |
|    time_elapsed    | 1455     |
|    total_timesteps | 72000    |
| train/             |          |
|    actor_loss      | -111     |
|    critic_loss     | 2.95     |
|    ent_coef        | 0.0522   |
|    ent_coef_loss   | -0.0257  |
|    learning_rate   | 0.0003   |
|    n_updates       | 71899    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 76       |
|    fps             | 48       |
|    time_elapsed    | 1564     |
|    total_timesteps | 76000    |
| train/             |          |
|    actor_loss      | -129     |
|    critic_loss     | 3.51     |
|    ent_coef        | 0.0549   |
|    ent_coef_loss   | 0.073    |
|    learning_rate   | 0.0003   |
|    n_updates       | 75899    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 80       |
|    fps             | 47       |
|    time_elapsed    | 1670     |
|    total_timesteps | 80000    |
| train/             |          |
|    actor_loss      | -134     |
|    critic_loss     | 3.41     |
|    ent_coef        | 0.0569   |
|    ent_coef_loss   | -0.173   |
|    learning_rate   | 0.0003   |
|    n_updates       | 79899    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 84       |
|    fps             | 47       |
|    time_elapsed    | 1777     |
|    total_timesteps | 84000    |
| train/             |          |
|    actor_loss      | -150     |
|    critic_loss     | 4.24     |
|    ent_coef        | 0.0595   |
|    ent_coef_loss   | -1.63    |
|    learning_rate   | 0.0003   |
|    n_updates       | 83899    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 88       |
|    fps             | 46       |
|    time_elapsed    | 1887     |
|    total_timesteps | 88000    |
| train/             |          |
|    actor_loss      | -169     |
|    critic_loss     | 4.42     |
|    ent_coef        | 0.0635   |
|    ent_coef_loss   | -1.48    |
|    learning_rate   | 0.0003   |
|    n_updates       | 87899    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 92       |
|    fps             | 46       |
|    time_elapsed    | 1992     |
|    total_timesteps | 92000    |
| train/             |          |
|    actor_loss      | -174     |
|    critic_loss     | 4.76     |
|    ent_coef        | 0.0674   |
|    ent_coef_loss   | -0.32    |
|    learning_rate   | 0.0003   |
|    n_updates       | 91899    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 96       |
|    fps             | 45       |
|    time_elapsed    | 2101     |
|    total_timesteps | 96000    |
| train/             |          |
|    actor_loss      | -180     |
|    critic_loss     | 4.35     |
|    ent_coef        | 0.0696   |
|    ent_coef_loss   | -1.46    |
|    learning_rate   | 0.0003   |
|    n_updates       | 95899    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 100      |
|    fps             | 45       |
|    time_elapsed    | 2207     |
|    total_timesteps | 100000   |
| train/             |          |
|    actor_loss      | -189     |
|    critic_loss     | 4.08     |
|    ent_coef        | 0.074    |
|    ent_coef_loss   | -0.294   |
|    learning_rate   | 0.0003   |
|    n_updates       | 99899    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 104      |
|    fps             | 44       |
|    time_elapsed    | 2316     |
|    total_timesteps | 104000   |
| train/             |          |
|    actor_loss      | -202     |
|    critic_loss     | 4.08     |
|    ent_coef        | 0.0761   |
|    ent_coef_loss   | -1.22    |
|    learning_rate   | 0.0003   |
|    n_updates       | 103899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 108      |
|    fps             | 44       |
|    time_elapsed    | 2419     |
|    total_timesteps | 108000   |
| train/             |          |
|    actor_loss      | -225     |
|    critic_loss     | 3.67     |
|    ent_coef        | 0.077    |
|    ent_coef_loss   | 0.366    |
|    learning_rate   | 0.0003   |
|    n_updates       | 107899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 112      |
|    fps             | 44       |
|    time_elapsed    | 2519     |
|    total_timesteps | 112000   |
| train/             |          |
|    actor_loss      | -225     |
|    critic_loss     | 4.32     |
|    ent_coef        | 0.0795   |
|    ent_coef_loss   | -0.241   |
|    learning_rate   | 0.0003   |
|    n_updates       | 111899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 116      |
|    fps             | 44       |
|    time_elapsed    | 2619     |
|    total_timesteps | 116000   |
| train/             |          |
|    actor_loss      | -228     |
|    critic_loss     | 3.72     |
|    ent_coef        | 0.0796   |
|    ent_coef_loss   | -0.471   |
|    learning_rate   | 0.0003   |
|    n_updates       | 115899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 120      |
|    fps             | 44       |
|    time_elapsed    | 2714     |
|    total_timesteps | 120000   |
| train/             |          |
|    actor_loss      | -232     |
|    critic_loss     | 3.53     |
|    ent_coef        | 0.0808   |
|    ent_coef_loss   | -0.694   |
|    learning_rate   | 0.0003   |
|    n_updates       | 119899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 124      |
|    fps             | 44       |
|    time_elapsed    | 2817     |
|    total_timesteps | 124000   |
| train/             |          |
|    actor_loss      | -244     |
|    critic_loss     | 4.69     |
|    ent_coef        | 0.0837   |
|    ent_coef_loss   | 0.81     |
|    learning_rate   | 0.0003   |
|    n_updates       | 123899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 128      |
|    fps             | 43       |
|    time_elapsed    | 2924     |
|    total_timesteps | 128000   |
| train/             |          |
|    actor_loss      | -246     |
|    critic_loss     | 2.92     |
|    ent_coef        | 0.086    |
|    ent_coef_loss   | -0.841   |
|    learning_rate   | 0.0003   |
|    n_updates       | 127899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 132      |
|    fps             | 43       |
|    time_elapsed    | 3033     |
|    total_timesteps | 132000   |
| train/             |          |
|    actor_loss      | -244     |
|    critic_loss     | 4.88     |
|    ent_coef        | 0.0851   |
|    ent_coef_loss   | -0.229   |
|    learning_rate   | 0.0003   |
|    n_updates       | 131899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 136      |
|    fps             | 43       |
|    time_elapsed    | 3131     |
|    total_timesteps | 136000   |
| train/             |          |
|    actor_loss      | -252     |
|    critic_loss     | 4.53     |
|    ent_coef        | 0.0868   |
|    ent_coef_loss   | 0.0654   |
|    learning_rate   | 0.0003   |
|    n_updates       | 135899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 140      |
|    fps             | 43       |
|    time_elapsed    | 3231     |
|    total_timesteps | 140000   |
| train/             |          |
|    actor_loss      | -255     |
|    critic_loss     | 3.95     |
|    ent_coef        | 0.0842   |
|    ent_coef_loss   | 0.243    |
|    learning_rate   | 0.0003   |
|    n_updates       | 139899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 144      |
|    fps             | 43       |
|    time_elapsed    | 3329     |
|    total_timesteps | 144000   |
| train/             |          |
|    actor_loss      | -272     |
|    critic_loss     | 3.45     |
|    ent_coef        | 0.088    |
|    ent_coef_loss   | -0.295   |
|    learning_rate   | 0.0003   |
|    n_updates       | 143899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 148      |
|    fps             | 43       |
|    time_elapsed    | 3426     |
|    total_timesteps | 148000   |
| train/             |          |
|    actor_loss      | -271     |
|    critic_loss     | 3.78     |
|    ent_coef        | 0.0853   |
|    ent_coef_loss   | -0.957   |
|    learning_rate   | 0.0003   |
|    n_updates       | 147899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 152      |
|    fps             | 43       |
|    time_elapsed    | 3529     |
|    total_timesteps | 152000   |
| train/             |          |
|    actor_loss      | -279     |
|    critic_loss     | 4.83     |
|    ent_coef        | 0.0875   |
|    ent_coef_loss   | 0.127    |
|    learning_rate   | 0.0003   |
|    n_updates       | 151899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 156      |
|    fps             | 42       |
|    time_elapsed    | 3629     |
|    total_timesteps | 156000   |
| train/             |          |
|    actor_loss      | -279     |
|    critic_loss     | 4.48     |
|    ent_coef        | 0.0894   |
|    ent_coef_loss   | 0.733    |
|    learning_rate   | 0.0003   |
|    n_updates       | 155899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 160      |
|    fps             | 42       |
|    time_elapsed    | 3726     |
|    total_timesteps | 160000   |
| train/             |          |
|    actor_loss      | -282     |
|    critic_loss     | 3.76     |
|    ent_coef        | 0.0911   |
|    ent_coef_loss   | -0.363   |
|    learning_rate   | 0.0003   |
|    n_updates       | 159899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 164      |
|    fps             | 42       |
|    time_elapsed    | 3823     |
|    total_timesteps | 164000   |
| train/             |          |
|    actor_loss      | -280     |
|    critic_loss     | 9.61     |
|    ent_coef        | 0.09     |
|    ent_coef_loss   | -0.979   |
|    learning_rate   | 0.0003   |
|    n_updates       | 163899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 168      |
|    fps             | 42       |
|    time_elapsed    | 3919     |
|    total_timesteps | 168000   |
| train/             |          |
|    actor_loss      | -299     |
|    critic_loss     | 4.71     |
|    ent_coef        | 0.0933   |
|    ent_coef_loss   | 1.61     |
|    learning_rate   | 0.0003   |
|    n_updates       | 167899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 172      |
|    fps             | 42       |
|    time_elapsed    | 4016     |
|    total_timesteps | 172000   |
| train/             |          |
|    actor_loss      | -309     |
|    critic_loss     | 3.47     |
|    ent_coef        | 0.092    |
|    ent_coef_loss   | 1.3      |
|    learning_rate   | 0.0003   |
|    n_updates       | 171899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 176      |
|    fps             | 42       |
|    time_elapsed    | 4113     |
|    total_timesteps | 176000   |
| train/             |          |
|    actor_loss      | -307     |
|    critic_loss     | 5.6      |
|    ent_coef        | 0.092    |
|    ent_coef_loss   | 0.661    |
|    learning_rate   | 0.0003   |
|    n_updates       | 175899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 180      |
|    fps             | 42       |
|    time_elapsed    | 4211     |
|    total_timesteps | 180000   |
| train/             |          |
|    actor_loss      | -307     |
|    critic_loss     | 3.6      |
|    ent_coef        | 0.094    |
|    ent_coef_loss   | -0.326   |
|    learning_rate   | 0.0003   |
|    n_updates       | 179899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 184      |
|    fps             | 42       |
|    time_elapsed    | 4310     |
|    total_timesteps | 184000   |
| train/             |          |
|    actor_loss      | -314     |
|    critic_loss     | 3.87     |
|    ent_coef        | 0.0949   |
|    ent_coef_loss   | -0.538   |
|    learning_rate   | 0.0003   |
|    n_updates       | 183899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 188      |
|    fps             | 42       |
|    time_elapsed    | 4408     |
|    total_timesteps | 188000   |
| train/             |          |
|    actor_loss      | -314     |
|    critic_loss     | 3.49     |
|    ent_coef        | 0.0961   |
|    ent_coef_loss   | -1.57    |
|    learning_rate   | 0.0003   |
|    n_updates       | 187899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 192      |
|    fps             | 42       |
|    time_elapsed    | 4508     |
|    total_timesteps | 192000   |
| train/             |          |
|    actor_loss      | -318     |
|    critic_loss     | 5.16     |
|    ent_coef        | 0.0966   |
|    ent_coef_loss   | 0.396    |
|    learning_rate   | 0.0003   |
|    n_updates       | 191899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 196      |
|    fps             | 42       |
|    time_elapsed    | 4608     |
|    total_timesteps | 196000   |
| train/             |          |
|    actor_loss      | -323     |
|    critic_loss     | 3.17     |
|    ent_coef        | 0.096    |
|    ent_coef_loss   | -0.168   |
|    learning_rate   | 0.0003   |
|    n_updates       | 195899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 200      |
|    fps             | 42       |
|    time_elapsed    | 4706     |
|    total_timesteps | 200000   |
| train/             |          |
|    actor_loss      | -333     |
|    critic_loss     | 4.44     |
|    ent_coef        | 0.0964   |
|    ent_coef_loss   | 0.173    |
|    learning_rate   | 0.0003   |
|    n_updates       | 199899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 204      |
|    fps             | 42       |
|    time_elapsed    | 4803     |
|    total_timesteps | 204000   |
| train/             |          |
|    actor_loss      | -337     |
|    critic_loss     | 4.7      |
|    ent_coef        | 0.0971   |
|    ent_coef_loss   | 1.36     |
|    learning_rate   | 0.0003   |
|    n_updates       | 203899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 208      |
|    fps             | 42       |
|    time_elapsed    | 4906     |
|    total_timesteps | 208000   |
| train/             |          |
|    actor_loss      | -331     |
|    critic_loss     | 3.86     |
|    ent_coef        | 0.0974   |
|    ent_coef_loss   | 0.918    |
|    learning_rate   | 0.0003   |
|    n_updates       | 207899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 212      |
|    fps             | 42       |
|    time_elapsed    | 5008     |
|    total_timesteps | 212000   |
| train/             |          |
|    actor_loss      | -341     |
|    critic_loss     | 3.75     |
|    ent_coef        | 0.0958   |
|    ent_coef_loss   | 0.00111  |
|    learning_rate   | 0.0003   |
|    n_updates       | 211899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 216      |
|    fps             | 42       |
|    time_elapsed    | 5105     |
|    total_timesteps | 216000   |
| train/             |          |
|    actor_loss      | -344     |
|    critic_loss     | 2.8      |
|    ent_coef        | 0.0984   |
|    ent_coef_loss   | 0.811    |
|    learning_rate   | 0.0003   |
|    n_updates       | 215899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 220      |
|    fps             | 42       |
|    time_elapsed    | 5202     |
|    total_timesteps | 220000   |
| train/             |          |
|    actor_loss      | -345     |
|    critic_loss     | 3.46     |
|    ent_coef        | 0.0985   |
|    ent_coef_loss   | -1.02    |
|    learning_rate   | 0.0003   |
|    n_updates       | 219899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 224      |
|    fps             | 42       |
|    time_elapsed    | 5298     |
|    total_timesteps | 224000   |
| train/             |          |
|    actor_loss      | -355     |
|    critic_loss     | 2.82     |
|    ent_coef        | 0.0989   |
|    ent_coef_loss   | -0.266   |
|    learning_rate   | 0.0003   |
|    n_updates       | 223899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 228      |
|    fps             | 42       |
|    time_elapsed    | 5405     |
|    total_timesteps | 228000   |
| train/             |          |
|    actor_loss      | -358     |
|    critic_loss     | 4.28     |
|    ent_coef        | 0.101    |
|    ent_coef_loss   | 0.362    |
|    learning_rate   | 0.0003   |
|    n_updates       | 227899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 232      |
|    fps             | 42       |
|    time_elapsed    | 5512     |
|    total_timesteps | 232000   |
| train/             |          |
|    actor_loss      | -372     |
|    critic_loss     | 3.89     |
|    ent_coef        | 0.102    |
|    ent_coef_loss   | -0.631   |
|    learning_rate   | 0.0003   |
|    n_updates       | 231899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 236      |
|    fps             | 41       |
|    time_elapsed    | 5619     |
|    total_timesteps | 236000   |
| train/             |          |
|    actor_loss      | -368     |
|    critic_loss     | 3.85     |
|    ent_coef        | 0.102    |
|    ent_coef_loss   | 0.968    |
|    learning_rate   | 0.0003   |
|    n_updates       | 235899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 240      |
|    fps             | 41       |
|    time_elapsed    | 5726     |
|    total_timesteps | 240000   |
| train/             |          |
|    actor_loss      | -367     |
|    critic_loss     | 4.45     |
|    ent_coef        | 0.103    |
|    ent_coef_loss   | -1.22    |
|    learning_rate   | 0.0003   |
|    n_updates       | 239899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 244      |
|    fps             | 41       |
|    time_elapsed    | 5831     |
|    total_timesteps | 244000   |
| train/             |          |
|    actor_loss      | -370     |
|    critic_loss     | 2.88     |
|    ent_coef        | 0.105    |
|    ent_coef_loss   | 0.676    |
|    learning_rate   | 0.0003   |
|    n_updates       | 243899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 248      |
|    fps             | 41       |
|    time_elapsed    | 5940     |
|    total_timesteps | 248000   |
| train/             |          |
|    actor_loss      | -384     |
|    critic_loss     | 4.15     |
|    ent_coef        | 0.104    |
|    ent_coef_loss   | 1.25     |
|    learning_rate   | 0.0003   |
|    n_updates       | 247899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 252      |
|    fps             | 41       |
|    time_elapsed    | 6049     |
|    total_timesteps | 252000   |
| train/             |          |
|    actor_loss      | -359     |
|    critic_loss     | 3.5      |
|    ent_coef        | 0.105    |
|    ent_coef_loss   | 0.137    |
|    learning_rate   | 0.0003   |
|    n_updates       | 251899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 256      |
|    fps             | 41       |
|    time_elapsed    | 6153     |
|    total_timesteps | 256000   |
| train/             |          |
|    actor_loss      | -376     |
|    critic_loss     | 3.56     |
|    ent_coef        | 0.107    |
|    ent_coef_loss   | -1.41    |
|    learning_rate   | 0.0003   |
|    n_updates       | 255899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 260      |
|    fps             | 41       |
|    time_elapsed    | 6259     |
|    total_timesteps | 260000   |
| train/             |          |
|    actor_loss      | -389     |
|    critic_loss     | 3.57     |
|    ent_coef        | 0.105    |
|    ent_coef_loss   | -0.0233  |
|    learning_rate   | 0.0003   |
|    n_updates       | 259899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 264      |
|    fps             | 41       |
|    time_elapsed    | 6366     |
|    total_timesteps | 264000   |
| train/             |          |
|    actor_loss      | -387     |
|    critic_loss     | 3.77     |
|    ent_coef        | 0.107    |
|    ent_coef_loss   | 2.35     |
|    learning_rate   | 0.0003   |
|    n_updates       | 263899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 268      |
|    fps             | 41       |
|    time_elapsed    | 6472     |
|    total_timesteps | 268000   |
| train/             |          |
|    actor_loss      | -391     |
|    critic_loss     | 3.38     |
|    ent_coef        | 0.106    |
|    ent_coef_loss   | -0.0966  |
|    learning_rate   | 0.0003   |
|    n_updates       | 267899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 272      |
|    fps             | 41       |
|    time_elapsed    | 6578     |
|    total_timesteps | 272000   |
| train/             |          |
|    actor_loss      | -395     |
|    critic_loss     | 4.38     |
|    ent_coef        | 0.109    |
|    ent_coef_loss   | -0.32    |
|    learning_rate   | 0.0003   |
|    n_updates       | 271899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 276      |
|    fps             | 41       |
|    time_elapsed    | 6685     |
|    total_timesteps | 276000   |
| train/             |          |
|    actor_loss      | -394     |
|    critic_loss     | 3.49     |
|    ent_coef        | 0.108    |
|    ent_coef_loss   | 0.163    |
|    learning_rate   | 0.0003   |
|    n_updates       | 275899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 280      |
|    fps             | 41       |
|    time_elapsed    | 6789     |
|    total_timesteps | 280000   |
| train/             |          |
|    actor_loss      | -410     |
|    critic_loss     | 4.53     |
|    ent_coef        | 0.109    |
|    ent_coef_loss   | 0.419    |
|    learning_rate   | 0.0003   |
|    n_updates       | 279899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 284      |
|    fps             | 41       |
|    time_elapsed    | 6897     |
|    total_timesteps | 284000   |
| train/             |          |
|    actor_loss      | -405     |
|    critic_loss     | 3.99     |
|    ent_coef        | 0.11     |
|    ent_coef_loss   | 0.0147   |
|    learning_rate   | 0.0003   |
|    n_updates       | 283899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 288      |
|    fps             | 41       |
|    time_elapsed    | 7007     |
|    total_timesteps | 288000   |
| train/             |          |
|    actor_loss      | -416     |
|    critic_loss     | 4.77     |
|    ent_coef        | 0.11     |
|    ent_coef_loss   | -0.246   |
|    learning_rate   | 0.0003   |
|    n_updates       | 287899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 292      |
|    fps             | 41       |
|    time_elapsed    | 7119     |
|    total_timesteps | 292000   |
| train/             |          |
|    actor_loss      | -405     |
|    critic_loss     | 3.63     |
|    ent_coef        | 0.11     |
|    ent_coef_loss   | -0.599   |
|    learning_rate   | 0.0003   |
|    n_updates       | 291899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 296      |
|    fps             | 40       |
|    time_elapsed    | 7225     |
|    total_timesteps | 296000   |
| train/             |          |
|    actor_loss      | -402     |
|    critic_loss     | 5.12     |
|    ent_coef        | 0.11     |
|    ent_coef_loss   | -1.01    |
|    learning_rate   | 0.0003   |
|    n_updates       | 295899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 300      |
|    fps             | 40       |
|    time_elapsed    | 7331     |
|    total_timesteps | 300000   |
| train/             |          |
|    actor_loss      | -421     |
|    critic_loss     | 5.47     |
|    ent_coef        | 0.111    |
|    ent_coef_loss   | -0.699   |
|    learning_rate   | 0.0003   |
|    n_updates       | 299899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 304      |
|    fps             | 40       |
|    time_elapsed    | 7438     |
|    total_timesteps | 304000   |
| train/             |          |
|    actor_loss      | -425     |
|    critic_loss     | 5.74     |
|    ent_coef        | 0.11     |
|    ent_coef_loss   | 0.379    |
|    learning_rate   | 0.0003   |
|    n_updates       | 303899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 308      |
|    fps             | 40       |
|    time_elapsed    | 7544     |
|    total_timesteps | 308000   |
| train/             |          |
|    actor_loss      | -429     |
|    critic_loss     | 5.1      |
|    ent_coef        | 0.112    |
|    ent_coef_loss   | 0.205    |
|    learning_rate   | 0.0003   |
|    n_updates       | 307899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 312      |
|    fps             | 40       |
|    time_elapsed    | 7651     |
|    total_timesteps | 312000   |
| train/             |          |
|    actor_loss      | -441     |
|    critic_loss     | 4.65     |
|    ent_coef        | 0.11     |
|    ent_coef_loss   | 1.29     |
|    learning_rate   | 0.0003   |
|    n_updates       | 311899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 316      |
|    fps             | 40       |
|    time_elapsed    | 7759     |
|    total_timesteps | 316000   |
| train/             |          |
|    actor_loss      | -459     |
|    critic_loss     | 4        |
|    ent_coef        | 0.114    |
|    ent_coef_loss   | 0.552    |
|    learning_rate   | 0.0003   |
|    n_updates       | 315899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 320      |
|    fps             | 40       |
|    time_elapsed    | 7852     |
|    total_timesteps | 320000   |
| train/             |          |
|    actor_loss      | -460     |
|    critic_loss     | 3.7      |
|    ent_coef        | 0.113    |
|    ent_coef_loss   | 0.728    |
|    learning_rate   | 0.0003   |
|    n_updates       | 319899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 324      |
|    fps             | 40       |
|    time_elapsed    | 7950     |
|    total_timesteps | 324000   |
| train/             |          |
|    actor_loss      | -451     |
|    critic_loss     | 4.43     |
|    ent_coef        | 0.113    |
|    ent_coef_loss   | -0.423   |
|    learning_rate   | 0.0003   |
|    n_updates       | 323899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 328      |
|    fps             | 40       |
|    time_elapsed    | 8046     |
|    total_timesteps | 328000   |
| train/             |          |
|    actor_loss      | -445     |
|    critic_loss     | 4.3      |
|    ent_coef        | 0.114    |
|    ent_coef_loss   | -0.511   |
|    learning_rate   | 0.0003   |
|    n_updates       | 327899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 332      |
|    fps             | 40       |
|    time_elapsed    | 8147     |
|    total_timesteps | 332000   |
| train/             |          |
|    actor_loss      | -452     |
|    critic_loss     | 5        |
|    ent_coef        | 0.114    |
|    ent_coef_loss   | -0.558   |
|    learning_rate   | 0.0003   |
|    n_updates       | 331899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 336      |
|    fps             | 40       |
|    time_elapsed    | 8244     |
|    total_timesteps | 336000   |
| train/             |          |
|    actor_loss      | -473     |
|    critic_loss     | 5.14     |
|    ent_coef        | 0.116    |
|    ent_coef_loss   | -0.0402  |
|    learning_rate   | 0.0003   |
|    n_updates       | 335899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 340      |
|    fps             | 40       |
|    time_elapsed    | 8345     |
|    total_timesteps | 340000   |
| train/             |          |
|    actor_loss      | -469     |
|    critic_loss     | 4        |
|    ent_coef        | 0.118    |
|    ent_coef_loss   | -0.275   |
|    learning_rate   | 0.0003   |
|    n_updates       | 339899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 344      |
|    fps             | 40       |
|    time_elapsed    | 8442     |
|    total_timesteps | 344000   |
| train/             |          |
|    actor_loss      | -479     |
|    critic_loss     | 4.27     |
|    ent_coef        | 0.116    |
|    ent_coef_loss   | -0.29    |
|    learning_rate   | 0.0003   |
|    n_updates       | 343899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 348      |
|    fps             | 40       |
|    time_elapsed    | 8538     |
|    total_timesteps | 348000   |
| train/             |          |
|    actor_loss      | -476     |
|    critic_loss     | 4.73     |
|    ent_coef        | 0.119    |
|    ent_coef_loss   | -0.187   |
|    learning_rate   | 0.0003   |
|    n_updates       | 347899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 352      |
|    fps             | 40       |
|    time_elapsed    | 8639     |
|    total_timesteps | 352000   |
| train/             |          |
|    actor_loss      | -488     |
|    critic_loss     | 4.38     |
|    ent_coef        | 0.122    |
|    ent_coef_loss   | 0.00903  |
|    learning_rate   | 0.0003   |
|    n_updates       | 351899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 356      |
|    fps             | 40       |
|    time_elapsed    | 8738     |
|    total_timesteps | 356000   |
| train/             |          |
|    actor_loss      | -481     |
|    critic_loss     | 5.6      |
|    ent_coef        | 0.123    |
|    ent_coef_loss   | 1.25     |
|    learning_rate   | 0.0003   |
|    n_updates       | 355899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 360      |
|    fps             | 40       |
|    time_elapsed    | 8832     |
|    total_timesteps | 360000   |
| train/             |          |
|    actor_loss      | -489     |
|    critic_loss     | 4.61     |
|    ent_coef        | 0.12     |
|    ent_coef_loss   | -0.166   |
|    learning_rate   | 0.0003   |
|    n_updates       | 359899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 364      |
|    fps             | 40       |
|    time_elapsed    | 8931     |
|    total_timesteps | 364000   |
| train/             |          |
|    actor_loss      | -511     |
|    critic_loss     | 4.92     |
|    ent_coef        | 0.123    |
|    ent_coef_loss   | 0.146    |
|    learning_rate   | 0.0003   |
|    n_updates       | 363899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 368      |
|    fps             | 40       |
|    time_elapsed    | 9026     |
|    total_timesteps | 368000   |
| train/             |          |
|    actor_loss      | -507     |
|    critic_loss     | 5.17     |
|    ent_coef        | 0.122    |
|    ent_coef_loss   | 0.239    |
|    learning_rate   | 0.0003   |
|    n_updates       | 367899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 372      |
|    fps             | 40       |
|    time_elapsed    | 9131     |
|    total_timesteps | 372000   |
| train/             |          |
|    actor_loss      | -502     |
|    critic_loss     | 5.43     |
|    ent_coef        | 0.124    |
|    ent_coef_loss   | -0.551   |
|    learning_rate   | 0.0003   |
|    n_updates       | 371899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 376      |
|    fps             | 40       |
|    time_elapsed    | 9225     |
|    total_timesteps | 376000   |
| train/             |          |
|    actor_loss      | -512     |
|    critic_loss     | 4.47     |
|    ent_coef        | 0.121    |
|    ent_coef_loss   | -0.519   |
|    learning_rate   | 0.0003   |
|    n_updates       | 375899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 380      |
|    fps             | 40       |
|    time_elapsed    | 9321     |
|    total_timesteps | 380000   |
| train/             |          |
|    actor_loss      | -520     |
|    critic_loss     | 4.25     |
|    ent_coef        | 0.121    |
|    ent_coef_loss   | -0.289   |
|    learning_rate   | 0.0003   |
|    n_updates       | 379899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 384      |
|    fps             | 40       |
|    time_elapsed    | 9420     |
|    total_timesteps | 384000   |
| train/             |          |
|    actor_loss      | -528     |
|    critic_loss     | 5.38     |
|    ent_coef        | 0.122    |
|    ent_coef_loss   | -0.0879  |
|    learning_rate   | 0.0003   |
|    n_updates       | 383899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 388      |
|    fps             | 40       |
|    time_elapsed    | 9526     |
|    total_timesteps | 388000   |
| train/             |          |
|    actor_loss      | -523     |
|    critic_loss     | 4.54     |
|    ent_coef        | 0.125    |
|    ent_coef_loss   | 0.689    |
|    learning_rate   | 0.0003   |
|    n_updates       | 387899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 392      |
|    fps             | 40       |
|    time_elapsed    | 9622     |
|    total_timesteps | 392000   |
| train/             |          |
|    actor_loss      | -525     |
|    critic_loss     | 5.35     |
|    ent_coef        | 0.125    |
|    ent_coef_loss   | -0.753   |
|    learning_rate   | 0.0003   |
|    n_updates       | 391899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 396      |
|    fps             | 40       |
|    time_elapsed    | 9723     |
|    total_timesteps | 396000   |
| train/             |          |
|    actor_loss      | -526     |
|    critic_loss     | 4.98     |
|    ent_coef        | 0.126    |
|    ent_coef_loss   | -0.525   |
|    learning_rate   | 0.0003   |
|    n_updates       | 395899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 400      |
|    fps             | 40       |
|    time_elapsed    | 9828     |
|    total_timesteps | 400000   |
| train/             |          |
|    actor_loss      | -539     |
|    critic_loss     | 5.56     |
|    ent_coef        | 0.127    |
|    ent_coef_loss   | -0.776   |
|    learning_rate   | 0.0003   |
|    n_updates       | 399899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 404      |
|    fps             | 40       |
|    time_elapsed    | 9929     |
|    total_timesteps | 404000   |
| train/             |          |
|    actor_loss      | -539     |
|    critic_loss     | 4.74     |
|    ent_coef        | 0.127    |
|    ent_coef_loss   | -0.272   |
|    learning_rate   | 0.0003   |
|    n_updates       | 403899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 408      |
|    fps             | 40       |
|    time_elapsed    | 10027    |
|    total_timesteps | 408000   |
| train/             |          |
|    actor_loss      | -534     |
|    critic_loss     | 6.54     |
|    ent_coef        | 0.127    |
|    ent_coef_loss   | 0.152    |
|    learning_rate   | 0.0003   |
|    n_updates       | 407899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 412      |
|    fps             | 40       |
|    time_elapsed    | 10130    |
|    total_timesteps | 412000   |
| train/             |          |
|    actor_loss      | -555     |
|    critic_loss     | 5.67     |
|    ent_coef        | 0.13     |
|    ent_coef_loss   | 0.0848   |
|    learning_rate   | 0.0003   |
|    n_updates       | 411899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 416      |
|    fps             | 40       |
|    time_elapsed    | 10225    |
|    total_timesteps | 416000   |
| train/             |          |
|    actor_loss      | -550     |
|    critic_loss     | 6.17     |
|    ent_coef        | 0.128    |
|    ent_coef_loss   | 0.511    |
|    learning_rate   | 0.0003   |
|    n_updates       | 415899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 420      |
|    fps             | 40       |
|    time_elapsed    | 10330    |
|    total_timesteps | 420000   |
| train/             |          |
|    actor_loss      | -559     |
|    critic_loss     | 5.88     |
|    ent_coef        | 0.128    |
|    ent_coef_loss   | -0.176   |
|    learning_rate   | 0.0003   |
|    n_updates       | 419899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 424      |
|    fps             | 40       |
|    time_elapsed    | 10432    |
|    total_timesteps | 424000   |
| train/             |          |
|    actor_loss      | -554     |
|    critic_loss     | 4.33     |
|    ent_coef        | 0.131    |
|    ent_coef_loss   | 0.124    |
|    learning_rate   | 0.0003   |
|    n_updates       | 423899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 428      |
|    fps             | 40       |
|    time_elapsed    | 10528    |
|    total_timesteps | 428000   |
| train/             |          |
|    actor_loss      | -560     |
|    critic_loss     | 6.84     |
|    ent_coef        | 0.131    |
|    ent_coef_loss   | -0.0304  |
|    learning_rate   | 0.0003   |
|    n_updates       | 427899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 432      |
|    fps             | 40       |
|    time_elapsed    | 10635    |
|    total_timesteps | 432000   |
| train/             |          |
|    actor_loss      | -566     |
|    critic_loss     | 7.08     |
|    ent_coef        | 0.135    |
|    ent_coef_loss   | -1.43    |
|    learning_rate   | 0.0003   |
|    n_updates       | 431899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 436      |
|    fps             | 40       |
|    time_elapsed    | 10744    |
|    total_timesteps | 436000   |
| train/             |          |
|    actor_loss      | -555     |
|    critic_loss     | 3.9      |
|    ent_coef        | 0.134    |
|    ent_coef_loss   | 0.0621   |
|    learning_rate   | 0.0003   |
|    n_updates       | 435899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 440      |
|    fps             | 40       |
|    time_elapsed    | 10850    |
|    total_timesteps | 440000   |
| train/             |          |
|    actor_loss      | -584     |
|    critic_loss     | 6.47     |
|    ent_coef        | 0.13     |
|    ent_coef_loss   | 0.933    |
|    learning_rate   | 0.0003   |
|    n_updates       | 439899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 444      |
|    fps             | 40       |
|    time_elapsed    | 10954    |
|    total_timesteps | 444000   |
| train/             |          |
|    actor_loss      | -580     |
|    critic_loss     | 8.28     |
|    ent_coef        | 0.135    |
|    ent_coef_loss   | 0.323    |
|    learning_rate   | 0.0003   |
|    n_updates       | 443899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 448      |
|    fps             | 40       |
|    time_elapsed    | 11060    |
|    total_timesteps | 448000   |
| train/             |          |
|    actor_loss      | -576     |
|    critic_loss     | 6.69     |
|    ent_coef        | 0.136    |
|    ent_coef_loss   | 0.203    |
|    learning_rate   | 0.0003   |
|    n_updates       | 447899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 452      |
|    fps             | 40       |
|    time_elapsed    | 11166    |
|    total_timesteps | 452000   |
| train/             |          |
|    actor_loss      | -585     |
|    critic_loss     | 4.9      |
|    ent_coef        | 0.134    |
|    ent_coef_loss   | -0.322   |
|    learning_rate   | 0.0003   |
|    n_updates       | 451899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 456      |
|    fps             | 40       |
|    time_elapsed    | 11273    |
|    total_timesteps | 456000   |
| train/             |          |
|    actor_loss      | -590     |
|    critic_loss     | 7        |
|    ent_coef        | 0.134    |
|    ent_coef_loss   | 0.146    |
|    learning_rate   | 0.0003   |
|    n_updates       | 455899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 460      |
|    fps             | 40       |
|    time_elapsed    | 11377    |
|    total_timesteps | 460000   |
| train/             |          |
|    actor_loss      | -588     |
|    critic_loss     | 5.98     |
|    ent_coef        | 0.136    |
|    ent_coef_loss   | 0.299    |
|    learning_rate   | 0.0003   |
|    n_updates       | 459899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 464      |
|    fps             | 40       |
|    time_elapsed    | 11481    |
|    total_timesteps | 464000   |
| train/             |          |
|    actor_loss      | -580     |
|    critic_loss     | 10.6     |
|    ent_coef        | 0.138    |
|    ent_coef_loss   | -0.452   |
|    learning_rate   | 0.0003   |
|    n_updates       | 463899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 468      |
|    fps             | 40       |
|    time_elapsed    | 11590    |
|    total_timesteps | 468000   |
| train/             |          |
|    actor_loss      | -599     |
|    critic_loss     | 4.81     |
|    ent_coef        | 0.139    |
|    ent_coef_loss   | -0.395   |
|    learning_rate   | 0.0003   |
|    n_updates       | 467899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 472      |
|    fps             | 40       |
|    time_elapsed    | 11697    |
|    total_timesteps | 472000   |
| train/             |          |
|    actor_loss      | -592     |
|    critic_loss     | 5.3      |
|    ent_coef        | 0.139    |
|    ent_coef_loss   | 1.64     |
|    learning_rate   | 0.0003   |
|    n_updates       | 471899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 476      |
|    fps             | 40       |
|    time_elapsed    | 11801    |
|    total_timesteps | 476000   |
| train/             |          |
|    actor_loss      | -599     |
|    critic_loss     | 6.24     |
|    ent_coef        | 0.139    |
|    ent_coef_loss   | -0.492   |
|    learning_rate   | 0.0003   |
|    n_updates       | 475899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 480      |
|    fps             | 40       |
|    time_elapsed    | 11906    |
|    total_timesteps | 480000   |
| train/             |          |
|    actor_loss      | -600     |
|    critic_loss     | 4.73     |
|    ent_coef        | 0.139    |
|    ent_coef_loss   | 0.154    |
|    learning_rate   | 0.0003   |
|    n_updates       | 479899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 484      |
|    fps             | 40       |
|    time_elapsed    | 12011    |
|    total_timesteps | 484000   |
| train/             |          |
|    actor_loss      | -600     |
|    critic_loss     | 5.31     |
|    ent_coef        | 0.142    |
|    ent_coef_loss   | 0.783    |
|    learning_rate   | 0.0003   |
|    n_updates       | 483899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 488      |
|    fps             | 40       |
|    time_elapsed    | 12117    |
|    total_timesteps | 488000   |
| train/             |          |
|    actor_loss      | -607     |
|    critic_loss     | 6.57     |
|    ent_coef        | 0.144    |
|    ent_coef_loss   | 0.538    |
|    learning_rate   | 0.0003   |
|    n_updates       | 487899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 492      |
|    fps             | 40       |
|    time_elapsed    | 12224    |
|    total_timesteps | 492000   |
| train/             |          |
|    actor_loss      | -608     |
|    critic_loss     | 5.67     |
|    ent_coef        | 0.141    |
|    ent_coef_loss   | -0.116   |
|    learning_rate   | 0.0003   |
|    n_updates       | 491899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 496      |
|    fps             | 40       |
|    time_elapsed    | 12330    |
|    total_timesteps | 496000   |
| train/             |          |
|    actor_loss      | -604     |
|    critic_loss     | 5.76     |
|    ent_coef        | 0.145    |
|    ent_coef_loss   | -0.309   |
|    learning_rate   | 0.0003   |
|    n_updates       | 495899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 500      |
|    fps             | 40       |
|    time_elapsed    | 12434    |
|    total_timesteps | 500000   |
| train/             |          |
|    actor_loss      | -607     |
|    critic_loss     | 4.65     |
|    ent_coef        | 0.144    |
|    ent_coef_loss   | 0.0128   |
|    learning_rate   | 0.0003   |
|    n_updates       | 499899   |
---------------------------------
/home/zhushaoq/.conda/envs/SB_FVRL/lib/python3.7/site-packages/glfw/__init__.py:916: GLFWError: (65544) b'X11: The DISPLAY environment variable is missing'
  warnings.warn(message, GLFWError)
/home/zhushaoq/.conda/envs/SB_FVRL/lib/python3.7/site-packages/torch/utils/tensorboard/__init__.py:5: DeprecationWarning: distutils Version classes are deprecated. Use packaging.version instead.
  tensorboard.__version__
/home/zhushaoq/.conda/envs/SB_FVRL/lib/python3.7/site-packages/torch/utils/tensorboard/__init__.py:6: DeprecationWarning: distutils Version classes are deprecated. Use packaging.version instead.
  ) < LooseVersion("1.15"):
/home/zhushaoq/.conda/envs/SB_FVRL/lib/python3.7/site-packages/gymnasium/envs/registration.py:524: DeprecationWarning: [33mWARN: The environment HalfCheetah-v3 is out of date. You should consider upgrading to version `v4`.[0m
  f"The environment {env_name} is out of date. You should consider "
/home/zhushaoq/.conda/envs/SB_FVRL/lib/python3.7/site-packages/gymnasium/envs/mujoco/mujoco_env.py:186: DeprecationWarning: [33mWARN: This version of the mujoco environments depends on the mujoco-py bindings, which are no longer maintained and may stop working. Please upgrade to the v4 versions of the environments (which depend on the mujoco python bindings instead), unless you are trying to precisely replicate previous works).[0m
  "This version of the mujoco environments depends "
