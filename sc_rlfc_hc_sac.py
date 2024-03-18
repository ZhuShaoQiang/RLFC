nohup: ignoring input
Using cuda device
None
None
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
|    fps             | 60       |
|    time_elapsed    | 66       |
|    total_timesteps | 4000     |
| train/             |          |
|    actor_loss      | -42      |
|    critic_loss     | 0.76     |
|    ent_coef        | 0.317    |
|    ent_coef_loss   | -10.2    |
|    learning_rate   | 0.0003   |
|    n_updates       | 3899     |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 8        |
|    fps             | 58       |
|    time_elapsed    | 136      |
|    total_timesteps | 8000     |
| train/             |          |
|    actor_loss      | -47.1    |
|    critic_loss     | 0.854    |
|    ent_coef        | 0.104    |
|    ent_coef_loss   | -13.5    |
|    learning_rate   | 0.0003   |
|    n_updates       | 7899     |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 12       |
|    fps             | 58       |
|    time_elapsed    | 205      |
|    total_timesteps | 12000    |
| train/             |          |
|    actor_loss      | -44.6    |
|    critic_loss     | 0.764    |
|    ent_coef        | 0.0375   |
|    ent_coef_loss   | -10.1    |
|    learning_rate   | 0.0003   |
|    n_updates       | 11899    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 16       |
|    fps             | 56       |
|    time_elapsed    | 281      |
|    total_timesteps | 16000    |
| train/             |          |
|    actor_loss      | -39.4    |
|    critic_loss     | 0.606    |
|    ent_coef        | 0.017    |
|    ent_coef_loss   | -3.35    |
|    learning_rate   | 0.0003   |
|    n_updates       | 15899    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 20       |
|    fps             | 57       |
|    time_elapsed    | 349      |
|    total_timesteps | 20000    |
| train/             |          |
|    actor_loss      | -34.1    |
|    critic_loss     | 0.487    |
|    ent_coef        | 0.0117   |
|    ent_coef_loss   | -1.81    |
|    learning_rate   | 0.0003   |
|    n_updates       | 19899    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 24       |
|    fps             | 56       |
|    time_elapsed    | 422      |
|    total_timesteps | 24000    |
| train/             |          |
|    actor_loss      | -28.6    |
|    critic_loss     | 0.465    |
|    ent_coef        | 0.00942  |
|    ent_coef_loss   | 0.0276   |
|    learning_rate   | 0.0003   |
|    n_updates       | 23899    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 28       |
|    fps             | 56       |
|    time_elapsed    | 496      |
|    total_timesteps | 28000    |
| train/             |          |
|    actor_loss      | -23.8    |
|    critic_loss     | 0.471    |
|    ent_coef        | 0.00809  |
|    ent_coef_loss   | 0.092    |
|    learning_rate   | 0.0003   |
|    n_updates       | 27899    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 32       |
|    fps             | 55       |
|    time_elapsed    | 572      |
|    total_timesteps | 32000    |
| train/             |          |
|    actor_loss      | -23.1    |
|    critic_loss     | 0.68     |
|    ent_coef        | 0.00957  |
|    ent_coef_loss   | -1.13    |
|    learning_rate   | 0.0003   |
|    n_updates       | 31899    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 36       |
|    fps             | 55       |
|    time_elapsed    | 644      |
|    total_timesteps | 36000    |
| train/             |          |
|    actor_loss      | -23.7    |
|    critic_loss     | 0.431    |
|    ent_coef        | 0.00876  |
|    ent_coef_loss   | -0.747   |
|    learning_rate   | 0.0003   |
|    n_updates       | 35899    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 40       |
|    fps             | 55       |
|    time_elapsed    | 726      |
|    total_timesteps | 40000    |
| train/             |          |
|    actor_loss      | -25.4    |
|    critic_loss     | 0.599    |
|    ent_coef        | 0.0103   |
|    ent_coef_loss   | -0.135   |
|    learning_rate   | 0.0003   |
|    n_updates       | 39899    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 44       |
|    fps             | 51       |
|    time_elapsed    | 847      |
|    total_timesteps | 44000    |
| train/             |          |
|    actor_loss      | -27.8    |
|    critic_loss     | 0.816    |
|    ent_coef        | 0.0163   |
|    ent_coef_loss   | 0.496    |
|    learning_rate   | 0.0003   |
|    n_updates       | 43899    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 48       |
|    fps             | 49       |
|    time_elapsed    | 967      |
|    total_timesteps | 48000    |
| train/             |          |
|    actor_loss      | -35.1    |
|    critic_loss     | 0.94     |
|    ent_coef        | 0.0249   |
|    ent_coef_loss   | -0.806   |
|    learning_rate   | 0.0003   |
|    n_updates       | 47899    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 52       |
|    fps             | 47       |
|    time_elapsed    | 1089     |
|    total_timesteps | 52000    |
| train/             |          |
|    actor_loss      | -46.8    |
|    critic_loss     | 1.18     |
|    ent_coef        | 0.0283   |
|    ent_coef_loss   | -2.08    |
|    learning_rate   | 0.0003   |
|    n_updates       | 51899    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 56       |
|    fps             | 46       |
|    time_elapsed    | 1215     |
|    total_timesteps | 56000    |
| train/             |          |
|    actor_loss      | -58      |
|    critic_loss     | 1.83     |
|    ent_coef        | 0.0347   |
|    ent_coef_loss   | 0.243    |
|    learning_rate   | 0.0003   |
|    n_updates       | 55899    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 60       |
|    fps             | 44       |
|    time_elapsed    | 1341     |
|    total_timesteps | 60000    |
| train/             |          |
|    actor_loss      | -71.6    |
|    critic_loss     | 2.39     |
|    ent_coef        | 0.0386   |
|    ent_coef_loss   | 0.243    |
|    learning_rate   | 0.0003   |
|    n_updates       | 59899    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 64       |
|    fps             | 43       |
|    time_elapsed    | 1464     |
|    total_timesteps | 64000    |
| train/             |          |
|    actor_loss      | -83.1    |
|    critic_loss     | 2.24     |
|    ent_coef        | 0.0407   |
|    ent_coef_loss   | -0.882   |
|    learning_rate   | 0.0003   |
|    n_updates       | 63899    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 68       |
|    fps             | 42       |
|    time_elapsed    | 1588     |
|    total_timesteps | 68000    |
| train/             |          |
|    actor_loss      | -92      |
|    critic_loss     | 3.23     |
|    ent_coef        | 0.0431   |
|    ent_coef_loss   | -1.45    |
|    learning_rate   | 0.0003   |
|    n_updates       | 67899    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 72       |
|    fps             | 42       |
|    time_elapsed    | 1708     |
|    total_timesteps | 72000    |
| train/             |          |
|    actor_loss      | -111     |
|    critic_loss     | 2.24     |
|    ent_coef        | 0.0475   |
|    ent_coef_loss   | -1.38    |
|    learning_rate   | 0.0003   |
|    n_updates       | 71899    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 76       |
|    fps             | 41       |
|    time_elapsed    | 1830     |
|    total_timesteps | 76000    |
| train/             |          |
|    actor_loss      | -130     |
|    critic_loss     | 2.66     |
|    ent_coef        | 0.0499   |
|    ent_coef_loss   | -0.266   |
|    learning_rate   | 0.0003   |
|    n_updates       | 75899    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 80       |
|    fps             | 40       |
|    time_elapsed    | 1952     |
|    total_timesteps | 80000    |
| train/             |          |
|    actor_loss      | -133     |
|    critic_loss     | 2.8      |
|    ent_coef        | 0.0501   |
|    ent_coef_loss   | -1.48    |
|    learning_rate   | 0.0003   |
|    n_updates       | 79899    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 84       |
|    fps             | 40       |
|    time_elapsed    | 2078     |
|    total_timesteps | 84000    |
| train/             |          |
|    actor_loss      | -150     |
|    critic_loss     | 3.6      |
|    ent_coef        | 0.053    |
|    ent_coef_loss   | -0.269   |
|    learning_rate   | 0.0003   |
|    n_updates       | 83899    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 88       |
|    fps             | 39       |
|    time_elapsed    | 2201     |
|    total_timesteps | 88000    |
| train/             |          |
|    actor_loss      | -166     |
|    critic_loss     | 3.33     |
|    ent_coef        | 0.0553   |
|    ent_coef_loss   | -0.097   |
|    learning_rate   | 0.0003   |
|    n_updates       | 87899    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 92       |
|    fps             | 39       |
|    time_elapsed    | 2325     |
|    total_timesteps | 92000    |
| train/             |          |
|    actor_loss      | -173     |
|    critic_loss     | 4        |
|    ent_coef        | 0.0566   |
|    ent_coef_loss   | 0.275    |
|    learning_rate   | 0.0003   |
|    n_updates       | 91899    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 96       |
|    fps             | 39       |
|    time_elapsed    | 2444     |
|    total_timesteps | 96000    |
| train/             |          |
|    actor_loss      | -175     |
|    critic_loss     | 3.63     |
|    ent_coef        | 0.0615   |
|    ent_coef_loss   | -1.32    |
|    learning_rate   | 0.0003   |
|    n_updates       | 95899    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 100      |
|    fps             | 39       |
|    time_elapsed    | 2554     |
|    total_timesteps | 100000   |
| train/             |          |
|    actor_loss      | -190     |
|    critic_loss     | 4.36     |
|    ent_coef        | 0.0633   |
|    ent_coef_loss   | -0.589   |
|    learning_rate   | 0.0003   |
|    n_updates       | 99899    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 104      |
|    fps             | 38       |
|    time_elapsed    | 2668     |
|    total_timesteps | 104000   |
| train/             |          |
|    actor_loss      | -207     |
|    critic_loss     | 3.46     |
|    ent_coef        | 0.0645   |
|    ent_coef_loss   | 0.317    |
|    learning_rate   | 0.0003   |
|    n_updates       | 103899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 108      |
|    fps             | 38       |
|    time_elapsed    | 2787     |
|    total_timesteps | 108000   |
| train/             |          |
|    actor_loss      | -220     |
|    critic_loss     | 6.27     |
|    ent_coef        | 0.0657   |
|    ent_coef_loss   | -0.458   |
|    learning_rate   | 0.0003   |
|    n_updates       | 107899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 112      |
|    fps             | 38       |
|    time_elapsed    | 2911     |
|    total_timesteps | 112000   |
| train/             |          |
|    actor_loss      | -219     |
|    critic_loss     | 3.7      |
|    ent_coef        | 0.0675   |
|    ent_coef_loss   | 0.841    |
|    learning_rate   | 0.0003   |
|    n_updates       | 111899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 116      |
|    fps             | 38       |
|    time_elapsed    | 3037     |
|    total_timesteps | 116000   |
| train/             |          |
|    actor_loss      | -222     |
|    critic_loss     | 4.26     |
|    ent_coef        | 0.07     |
|    ent_coef_loss   | -0.129   |
|    learning_rate   | 0.0003   |
|    n_updates       | 115899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 120      |
|    fps             | 38       |
|    time_elapsed    | 3152     |
|    total_timesteps | 120000   |
| train/             |          |
|    actor_loss      | -233     |
|    critic_loss     | 4.69     |
|    ent_coef        | 0.0724   |
|    ent_coef_loss   | 0.546    |
|    learning_rate   | 0.0003   |
|    n_updates       | 119899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 124      |
|    fps             | 37       |
|    time_elapsed    | 3269     |
|    total_timesteps | 124000   |
| train/             |          |
|    actor_loss      | -240     |
|    critic_loss     | 6.51     |
|    ent_coef        | 0.0734   |
|    ent_coef_loss   | 1.56     |
|    learning_rate   | 0.0003   |
|    n_updates       | 123899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 128      |
|    fps             | 37       |
|    time_elapsed    | 3384     |
|    total_timesteps | 128000   |
| train/             |          |
|    actor_loss      | -238     |
|    critic_loss     | 7.09     |
|    ent_coef        | 0.0758   |
|    ent_coef_loss   | -0.901   |
|    learning_rate   | 0.0003   |
|    n_updates       | 127899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 132      |
|    fps             | 37       |
|    time_elapsed    | 3504     |
|    total_timesteps | 132000   |
| train/             |          |
|    actor_loss      | -236     |
|    critic_loss     | 8.2      |
|    ent_coef        | 0.0787   |
|    ent_coef_loss   | -1.71    |
|    learning_rate   | 0.0003   |
|    n_updates       | 131899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 136      |
|    fps             | 37       |
|    time_elapsed    | 3620     |
|    total_timesteps | 136000   |
| train/             |          |
|    actor_loss      | -247     |
|    critic_loss     | 5.51     |
|    ent_coef        | 0.0782   |
|    ent_coef_loss   | 0.111    |
|    learning_rate   | 0.0003   |
|    n_updates       | 135899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 140      |
|    fps             | 37       |
|    time_elapsed    | 3735     |
|    total_timesteps | 140000   |
| train/             |          |
|    actor_loss      | -251     |
|    critic_loss     | 6.08     |
|    ent_coef        | 0.0779   |
|    ent_coef_loss   | 0.654    |
|    learning_rate   | 0.0003   |
|    n_updates       | 139899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 144      |
|    fps             | 37       |
|    time_elapsed    | 3849     |
|    total_timesteps | 144000   |
| train/             |          |
|    actor_loss      | -280     |
|    critic_loss     | 5.69     |
|    ent_coef        | 0.0732   |
|    ent_coef_loss   | 0.802    |
|    learning_rate   | 0.0003   |
|    n_updates       | 143899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 148      |
|    fps             | 37       |
|    time_elapsed    | 3966     |
|    total_timesteps | 148000   |
| train/             |          |
|    actor_loss      | -267     |
|    critic_loss     | 7.27     |
|    ent_coef        | 0.0769   |
|    ent_coef_loss   | 1.34     |
|    learning_rate   | 0.0003   |
|    n_updates       | 147899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 152      |
|    fps             | 37       |
|    time_elapsed    | 4080     |
|    total_timesteps | 152000   |
| train/             |          |
|    actor_loss      | -271     |
|    critic_loss     | 4.34     |
|    ent_coef        | 0.082    |
|    ent_coef_loss   | 0.505    |
|    learning_rate   | 0.0003   |
|    n_updates       | 151899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 156      |
|    fps             | 37       |
|    time_elapsed    | 4193     |
|    total_timesteps | 156000   |
| train/             |          |
|    actor_loss      | -273     |
|    critic_loss     | 7.15     |
|    ent_coef        | 0.0866   |
|    ent_coef_loss   | -0.966   |
|    learning_rate   | 0.0003   |
|    n_updates       | 155899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 160      |
|    fps             | 37       |
|    time_elapsed    | 4309     |
|    total_timesteps | 160000   |
| train/             |          |
|    actor_loss      | -281     |
|    critic_loss     | 4.98     |
|    ent_coef        | 0.0855   |
|    ent_coef_loss   | 0.385    |
|    learning_rate   | 0.0003   |
|    n_updates       | 159899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 164      |
|    fps             | 37       |
|    time_elapsed    | 4424     |
|    total_timesteps | 164000   |
| train/             |          |
|    actor_loss      | -284     |
|    critic_loss     | 5.32     |
|    ent_coef        | 0.0878   |
|    ent_coef_loss   | 0.0364   |
|    learning_rate   | 0.0003   |
|    n_updates       | 163899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 168      |
|    fps             | 36       |
|    time_elapsed    | 4541     |
|    total_timesteps | 168000   |
| train/             |          |
|    actor_loss      | -307     |
|    critic_loss     | 4.33     |
|    ent_coef        | 0.09     |
|    ent_coef_loss   | 0.856    |
|    learning_rate   | 0.0003   |
|    n_updates       | 167899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 172      |
|    fps             | 36       |
|    time_elapsed    | 4657     |
|    total_timesteps | 172000   |
| train/             |          |
|    actor_loss      | -297     |
|    critic_loss     | 6.86     |
|    ent_coef        | 0.0902   |
|    ent_coef_loss   | 0.491    |
|    learning_rate   | 0.0003   |
|    n_updates       | 171899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 176      |
|    fps             | 36       |
|    time_elapsed    | 4774     |
|    total_timesteps | 176000   |
| train/             |          |
|    actor_loss      | -304     |
|    critic_loss     | 6.46     |
|    ent_coef        | 0.0915   |
|    ent_coef_loss   | 1.08     |
|    learning_rate   | 0.0003   |
|    n_updates       | 175899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 180      |
|    fps             | 36       |
|    time_elapsed    | 4892     |
|    total_timesteps | 180000   |
| train/             |          |
|    actor_loss      | -304     |
|    critic_loss     | 5.03     |
|    ent_coef        | 0.0896   |
|    ent_coef_loss   | -1.05    |
|    learning_rate   | 0.0003   |
|    n_updates       | 179899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 184      |
|    fps             | 36       |
|    time_elapsed    | 5011     |
|    total_timesteps | 184000   |
| train/             |          |
|    actor_loss      | -305     |
|    critic_loss     | 3.59     |
|    ent_coef        | 0.0922   |
|    ent_coef_loss   | 0.726    |
|    learning_rate   | 0.0003   |
|    n_updates       | 183899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 188      |
|    fps             | 36       |
|    time_elapsed    | 5125     |
|    total_timesteps | 188000   |
| train/             |          |
|    actor_loss      | -300     |
|    critic_loss     | 4.36     |
|    ent_coef        | 0.0942   |
|    ent_coef_loss   | -0.0986  |
|    learning_rate   | 0.0003   |
|    n_updates       | 187899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 192      |
|    fps             | 36       |
|    time_elapsed    | 5237     |
|    total_timesteps | 192000   |
| train/             |          |
|    actor_loss      | -307     |
|    critic_loss     | 9.03     |
|    ent_coef        | 0.0917   |
|    ent_coef_loss   | 0.247    |
|    learning_rate   | 0.0003   |
|    n_updates       | 191899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 196      |
|    fps             | 36       |
|    time_elapsed    | 5359     |
|    total_timesteps | 196000   |
| train/             |          |
|    actor_loss      | -311     |
|    critic_loss     | 4.69     |
|    ent_coef        | 0.0922   |
|    ent_coef_loss   | -0.845   |
|    learning_rate   | 0.0003   |
|    n_updates       | 195899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 200      |
|    fps             | 36       |
|    time_elapsed    | 5482     |
|    total_timesteps | 200000   |
| train/             |          |
|    actor_loss      | -306     |
|    critic_loss     | 3.77     |
|    ent_coef        | 0.0943   |
|    ent_coef_loss   | -0.814   |
|    learning_rate   | 0.0003   |
|    n_updates       | 199899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 204      |
|    fps             | 36       |
|    time_elapsed    | 5604     |
|    total_timesteps | 204000   |
| train/             |          |
|    actor_loss      | -315     |
|    critic_loss     | 6.24     |
|    ent_coef        | 0.0937   |
|    ent_coef_loss   | -0.284   |
|    learning_rate   | 0.0003   |
|    n_updates       | 203899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 208      |
|    fps             | 36       |
|    time_elapsed    | 5727     |
|    total_timesteps | 208000   |
| train/             |          |
|    actor_loss      | -312     |
|    critic_loss     | 4.05     |
|    ent_coef        | 0.0955   |
|    ent_coef_loss   | -0.484   |
|    learning_rate   | 0.0003   |
|    n_updates       | 207899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 212      |
|    fps             | 36       |
|    time_elapsed    | 5847     |
|    total_timesteps | 212000   |
| train/             |          |
|    actor_loss      | -328     |
|    critic_loss     | 3.64     |
|    ent_coef        | 0.0967   |
|    ent_coef_loss   | -0.243   |
|    learning_rate   | 0.0003   |
|    n_updates       | 211899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 216      |
|    fps             | 36       |
|    time_elapsed    | 5973     |
|    total_timesteps | 216000   |
| train/             |          |
|    actor_loss      | -320     |
|    critic_loss     | 2.98     |
|    ent_coef        | 0.0981   |
|    ent_coef_loss   | -0.121   |
|    learning_rate   | 0.0003   |
|    n_updates       | 215899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 220      |
|    fps             | 36       |
|    time_elapsed    | 6098     |
|    total_timesteps | 220000   |
| train/             |          |
|    actor_loss      | -321     |
|    critic_loss     | 9.08     |
|    ent_coef        | 0.0986   |
|    ent_coef_loss   | 0.392    |
|    learning_rate   | 0.0003   |
|    n_updates       | 219899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 224      |
|    fps             | 36       |
|    time_elapsed    | 6220     |
|    total_timesteps | 224000   |
| train/             |          |
|    actor_loss      | -323     |
|    critic_loss     | 2.36     |
|    ent_coef        | 0.101    |
|    ent_coef_loss   | -1.11    |
|    learning_rate   | 0.0003   |
|    n_updates       | 223899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 228      |
|    fps             | 35       |
|    time_elapsed    | 6343     |
|    total_timesteps | 228000   |
| train/             |          |
|    actor_loss      | -338     |
|    critic_loss     | 3.73     |
|    ent_coef        | 0.0981   |
|    ent_coef_loss   | 0.54     |
|    learning_rate   | 0.0003   |
|    n_updates       | 227899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 232      |
|    fps             | 35       |
|    time_elapsed    | 6466     |
|    total_timesteps | 232000   |
| train/             |          |
|    actor_loss      | -329     |
|    critic_loss     | 3.26     |
|    ent_coef        | 0.101    |
|    ent_coef_loss   | -1       |
|    learning_rate   | 0.0003   |
|    n_updates       | 231899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 236      |
|    fps             | 35       |
|    time_elapsed    | 6588     |
|    total_timesteps | 236000   |
| train/             |          |
|    actor_loss      | -333     |
|    critic_loss     | 2.72     |
|    ent_coef        | 0.099    |
|    ent_coef_loss   | -0.0242  |
|    learning_rate   | 0.0003   |
|    n_updates       | 235899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 240      |
|    fps             | 35       |
|    time_elapsed    | 6710     |
|    total_timesteps | 240000   |
| train/             |          |
|    actor_loss      | -337     |
|    critic_loss     | 3.09     |
|    ent_coef        | 0.1      |
|    ent_coef_loss   | 1.86     |
|    learning_rate   | 0.0003   |
|    n_updates       | 239899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 244      |
|    fps             | 35       |
|    time_elapsed    | 6834     |
|    total_timesteps | 244000   |
| train/             |          |
|    actor_loss      | -333     |
|    critic_loss     | 4.59     |
|    ent_coef        | 0.101    |
|    ent_coef_loss   | 1.14     |
|    learning_rate   | 0.0003   |
|    n_updates       | 243899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 248      |
|    fps             | 35       |
|    time_elapsed    | 6958     |
|    total_timesteps | 248000   |
| train/             |          |
|    actor_loss      | -339     |
|    critic_loss     | 3.81     |
|    ent_coef        | 0.101    |
|    ent_coef_loss   | 0.972    |
|    learning_rate   | 0.0003   |
|    n_updates       | 247899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 252      |
|    fps             | 35       |
|    time_elapsed    | 7085     |
|    total_timesteps | 252000   |
| train/             |          |
|    actor_loss      | -333     |
|    critic_loss     | 3.85     |
|    ent_coef        | 0.103    |
|    ent_coef_loss   | -0.052   |
|    learning_rate   | 0.0003   |
|    n_updates       | 251899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 256      |
|    fps             | 35       |
|    time_elapsed    | 7207     |
|    total_timesteps | 256000   |
| train/             |          |
|    actor_loss      | -337     |
|    critic_loss     | 2.7      |
|    ent_coef        | 0.105    |
|    ent_coef_loss   | -0.469   |
|    learning_rate   | 0.0003   |
|    n_updates       | 255899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 260      |
|    fps             | 35       |
|    time_elapsed    | 7331     |
|    total_timesteps | 260000   |
| train/             |          |
|    actor_loss      | -352     |
|    critic_loss     | 5.49     |
|    ent_coef        | 0.103    |
|    ent_coef_loss   | 0.367    |
|    learning_rate   | 0.0003   |
|    n_updates       | 259899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 264      |
|    fps             | 35       |
|    time_elapsed    | 7452     |
|    total_timesteps | 264000   |
| train/             |          |
|    actor_loss      | -328     |
|    critic_loss     | 2.51     |
|    ent_coef        | 0.104    |
|    ent_coef_loss   | -1.06    |
|    learning_rate   | 0.0003   |
|    n_updates       | 263899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 268      |
|    fps             | 35       |
|    time_elapsed    | 7578     |
|    total_timesteps | 268000   |
| train/             |          |
|    actor_loss      | -348     |
|    critic_loss     | 2.97     |
|    ent_coef        | 0.105    |
|    ent_coef_loss   | -0.194   |
|    learning_rate   | 0.0003   |
|    n_updates       | 267899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 272      |
|    fps             | 35       |
|    time_elapsed    | 7698     |
|    total_timesteps | 272000   |
| train/             |          |
|    actor_loss      | -362     |
|    critic_loss     | 2.33     |
|    ent_coef        | 0.106    |
|    ent_coef_loss   | 1.85     |
|    learning_rate   | 0.0003   |
|    n_updates       | 271899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 276      |
|    fps             | 35       |
|    time_elapsed    | 7809     |
|    total_timesteps | 276000   |
| train/             |          |
|    actor_loss      | -358     |
|    critic_loss     | 3.12     |
|    ent_coef        | 0.107    |
|    ent_coef_loss   | 0.605    |
|    learning_rate   | 0.0003   |
|    n_updates       | 275899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 280      |
|    fps             | 35       |
|    time_elapsed    | 7924     |
|    total_timesteps | 280000   |
| train/             |          |
|    actor_loss      | -357     |
|    critic_loss     | 3.26     |
|    ent_coef        | 0.105    |
|    ent_coef_loss   | -0.395   |
|    learning_rate   | 0.0003   |
|    n_updates       | 279899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 284      |
|    fps             | 35       |
|    time_elapsed    | 8037     |
|    total_timesteps | 284000   |
| train/             |          |
|    actor_loss      | -353     |
|    critic_loss     | 3.34     |
|    ent_coef        | 0.107    |
|    ent_coef_loss   | -0.478   |
|    learning_rate   | 0.0003   |
|    n_updates       | 283899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 288      |
|    fps             | 35       |
|    time_elapsed    | 8154     |
|    total_timesteps | 288000   |
| train/             |          |
|    actor_loss      | -363     |
|    critic_loss     | 4.38     |
|    ent_coef        | 0.107    |
|    ent_coef_loss   | 0.636    |
|    learning_rate   | 0.0003   |
|    n_updates       | 287899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 292      |
|    fps             | 35       |
|    time_elapsed    | 8269     |
|    total_timesteps | 292000   |
| train/             |          |
|    actor_loss      | -350     |
|    critic_loss     | 2.96     |
|    ent_coef        | 0.108    |
|    ent_coef_loss   | -0.561   |
|    learning_rate   | 0.0003   |
|    n_updates       | 291899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 296      |
|    fps             | 35       |
|    time_elapsed    | 8386     |
|    total_timesteps | 296000   |
| train/             |          |
|    actor_loss      | -341     |
|    critic_loss     | 2.89     |
|    ent_coef        | 0.109    |
|    ent_coef_loss   | -1.06    |
|    learning_rate   | 0.0003   |
|    n_updates       | 295899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 300      |
|    fps             | 35       |
|    time_elapsed    | 8502     |
|    total_timesteps | 300000   |
| train/             |          |
|    actor_loss      | -362     |
|    critic_loss     | 3.4      |
|    ent_coef        | 0.109    |
|    ent_coef_loss   | 0.027    |
|    learning_rate   | 0.0003   |
|    n_updates       | 299899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 304      |
|    fps             | 35       |
|    time_elapsed    | 8621     |
|    total_timesteps | 304000   |
| train/             |          |
|    actor_loss      | -374     |
|    critic_loss     | 3.79     |
|    ent_coef        | 0.11     |
|    ent_coef_loss   | 0.672    |
|    learning_rate   | 0.0003   |
|    n_updates       | 303899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 308      |
|    fps             | 35       |
|    time_elapsed    | 8735     |
|    total_timesteps | 308000   |
| train/             |          |
|    actor_loss      | -371     |
|    critic_loss     | 3.5      |
|    ent_coef        | 0.108    |
|    ent_coef_loss   | -0.316   |
|    learning_rate   | 0.0003   |
|    n_updates       | 307899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 312      |
|    fps             | 35       |
|    time_elapsed    | 8847     |
|    total_timesteps | 312000   |
| train/             |          |
|    actor_loss      | -369     |
|    critic_loss     | 4.31     |
|    ent_coef        | 0.108    |
|    ent_coef_loss   | 0.232    |
|    learning_rate   | 0.0003   |
|    n_updates       | 311899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 316      |
|    fps             | 35       |
|    time_elapsed    | 8960     |
|    total_timesteps | 316000   |
| train/             |          |
|    actor_loss      | -372     |
|    critic_loss     | 3.48     |
|    ent_coef        | 0.108    |
|    ent_coef_loss   | 1.06     |
|    learning_rate   | 0.0003   |
|    n_updates       | 315899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 320      |
|    fps             | 35       |
|    time_elapsed    | 9082     |
|    total_timesteps | 320000   |
| train/             |          |
|    actor_loss      | -371     |
|    critic_loss     | 2.71     |
|    ent_coef        | 0.11     |
|    ent_coef_loss   | -0.618   |
|    learning_rate   | 0.0003   |
|    n_updates       | 319899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 324      |
|    fps             | 35       |
|    time_elapsed    | 9193     |
|    total_timesteps | 324000   |
| train/             |          |
|    actor_loss      | -390     |
|    critic_loss     | 3.46     |
|    ent_coef        | 0.111    |
|    ent_coef_loss   | 1.46     |
|    learning_rate   | 0.0003   |
|    n_updates       | 323899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 328      |
|    fps             | 35       |
|    time_elapsed    | 9307     |
|    total_timesteps | 328000   |
| train/             |          |
|    actor_loss      | -359     |
|    critic_loss     | 3.62     |
|    ent_coef        | 0.109    |
|    ent_coef_loss   | -0.484   |
|    learning_rate   | 0.0003   |
|    n_updates       | 327899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 332      |
|    fps             | 35       |
|    time_elapsed    | 9427     |
|    total_timesteps | 332000   |
| train/             |          |
|    actor_loss      | -376     |
|    critic_loss     | 2.29     |
|    ent_coef        | 0.109    |
|    ent_coef_loss   | 0.363    |
|    learning_rate   | 0.0003   |
|    n_updates       | 331899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 336      |
|    fps             | 35       |
|    time_elapsed    | 9547     |
|    total_timesteps | 336000   |
| train/             |          |
|    actor_loss      | -386     |
|    critic_loss     | 2.18     |
|    ent_coef        | 0.112    |
|    ent_coef_loss   | 0.193    |
|    learning_rate   | 0.0003   |
|    n_updates       | 335899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 340      |
|    fps             | 35       |
|    time_elapsed    | 9661     |
|    total_timesteps | 340000   |
| train/             |          |
|    actor_loss      | -378     |
|    critic_loss     | 4.17     |
|    ent_coef        | 0.11     |
|    ent_coef_loss   | -0.485   |
|    learning_rate   | 0.0003   |
|    n_updates       | 339899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 344      |
|    fps             | 35       |
|    time_elapsed    | 9781     |
|    total_timesteps | 344000   |
| train/             |          |
|    actor_loss      | -378     |
|    critic_loss     | 3        |
|    ent_coef        | 0.112    |
|    ent_coef_loss   | -0.302   |
|    learning_rate   | 0.0003   |
|    n_updates       | 343899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 348      |
|    fps             | 35       |
|    time_elapsed    | 9899     |
|    total_timesteps | 348000   |
| train/             |          |
|    actor_loss      | -373     |
|    critic_loss     | 4.55     |
|    ent_coef        | 0.112    |
|    ent_coef_loss   | -0.674   |
|    learning_rate   | 0.0003   |
|    n_updates       | 347899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 352      |
|    fps             | 35       |
|    time_elapsed    | 10013    |
|    total_timesteps | 352000   |
| train/             |          |
|    actor_loss      | -383     |
|    critic_loss     | 3.2      |
|    ent_coef        | 0.112    |
|    ent_coef_loss   | -0.988   |
|    learning_rate   | 0.0003   |
|    n_updates       | 351899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 356      |
|    fps             | 35       |
|    time_elapsed    | 10134    |
|    total_timesteps | 356000   |
| train/             |          |
|    actor_loss      | -388     |
|    critic_loss     | 4.13     |
|    ent_coef        | 0.113    |
|    ent_coef_loss   | -0.26    |
|    learning_rate   | 0.0003   |
|    n_updates       | 355899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 360      |
|    fps             | 35       |
|    time_elapsed    | 10252    |
|    total_timesteps | 360000   |
| train/             |          |
|    actor_loss      | -377     |
|    critic_loss     | 3.75     |
|    ent_coef        | 0.113    |
|    ent_coef_loss   | -0.688   |
|    learning_rate   | 0.0003   |
|    n_updates       | 359899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 364      |
|    fps             | 35       |
|    time_elapsed    | 10373    |
|    total_timesteps | 364000   |
| train/             |          |
|    actor_loss      | -401     |
|    critic_loss     | 4.43     |
|    ent_coef        | 0.112    |
|    ent_coef_loss   | 0.16     |
|    learning_rate   | 0.0003   |
|    n_updates       | 363899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 368      |
|    fps             | 35       |
|    time_elapsed    | 10487    |
|    total_timesteps | 368000   |
| train/             |          |
|    actor_loss      | -395     |
|    critic_loss     | 5.17     |
|    ent_coef        | 0.114    |
|    ent_coef_loss   | 0.025    |
|    learning_rate   | 0.0003   |
|    n_updates       | 367899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 372      |
|    fps             | 35       |
|    time_elapsed    | 10610    |
|    total_timesteps | 372000   |
| train/             |          |
|    actor_loss      | -382     |
|    critic_loss     | 3.33     |
|    ent_coef        | 0.113    |
|    ent_coef_loss   | -0.321   |
|    learning_rate   | 0.0003   |
|    n_updates       | 371899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 376      |
|    fps             | 35       |
|    time_elapsed    | 10733    |
|    total_timesteps | 376000   |
| train/             |          |
|    actor_loss      | -393     |
|    critic_loss     | 2.45     |
|    ent_coef        | 0.114    |
|    ent_coef_loss   | -0.667   |
|    learning_rate   | 0.0003   |
|    n_updates       | 375899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 380      |
|    fps             | 35       |
|    time_elapsed    | 10855    |
|    total_timesteps | 380000   |
| train/             |          |
|    actor_loss      | -385     |
|    critic_loss     | 2.66     |
|    ent_coef        | 0.116    |
|    ent_coef_loss   | -0.405   |
|    learning_rate   | 0.0003   |
|    n_updates       | 379899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 384      |
|    fps             | 34       |
|    time_elapsed    | 10978    |
|    total_timesteps | 384000   |
| train/             |          |
|    actor_loss      | -381     |
|    critic_loss     | 5        |
|    ent_coef        | 0.114    |
|    ent_coef_loss   | -0.172   |
|    learning_rate   | 0.0003   |
|    n_updates       | 383899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 388      |
|    fps             | 34       |
|    time_elapsed    | 11101    |
|    total_timesteps | 388000   |
| train/             |          |
|    actor_loss      | -397     |
|    critic_loss     | 4.97     |
|    ent_coef        | 0.114    |
|    ent_coef_loss   | 0.364    |
|    learning_rate   | 0.0003   |
|    n_updates       | 387899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 392      |
|    fps             | 34       |
|    time_elapsed    | 11225    |
|    total_timesteps | 392000   |
| train/             |          |
|    actor_loss      | -385     |
|    critic_loss     | 10.2     |
|    ent_coef        | 0.116    |
|    ent_coef_loss   | -0.0134  |
|    learning_rate   | 0.0003   |
|    n_updates       | 391899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 396      |
|    fps             | 34       |
|    time_elapsed    | 11344    |
|    total_timesteps | 396000   |
| train/             |          |
|    actor_loss      | -398     |
|    critic_loss     | 2.57     |
|    ent_coef        | 0.115    |
|    ent_coef_loss   | -1.41    |
|    learning_rate   | 0.0003   |
|    n_updates       | 395899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 400      |
|    fps             | 34       |
|    time_elapsed    | 11463    |
|    total_timesteps | 400000   |
| train/             |          |
|    actor_loss      | -397     |
|    critic_loss     | 3.68     |
|    ent_coef        | 0.116    |
|    ent_coef_loss   | 1.02     |
|    learning_rate   | 0.0003   |
|    n_updates       | 399899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 404      |
|    fps             | 34       |
|    time_elapsed    | 11588    |
|    total_timesteps | 404000   |
| train/             |          |
|    actor_loss      | -395     |
|    critic_loss     | 2.63     |
|    ent_coef        | 0.116    |
|    ent_coef_loss   | -1.24    |
|    learning_rate   | 0.0003   |
|    n_updates       | 403899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 408      |
|    fps             | 34       |
|    time_elapsed    | 11710    |
|    total_timesteps | 408000   |
| train/             |          |
|    actor_loss      | -397     |
|    critic_loss     | 4.68     |
|    ent_coef        | 0.119    |
|    ent_coef_loss   | -0.778   |
|    learning_rate   | 0.0003   |
|    n_updates       | 407899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 412      |
|    fps             | 34       |
|    time_elapsed    | 11831    |
|    total_timesteps | 412000   |
| train/             |          |
|    actor_loss      | -403     |
|    critic_loss     | 4.17     |
|    ent_coef        | 0.116    |
|    ent_coef_loss   | 0.0298   |
|    learning_rate   | 0.0003   |
|    n_updates       | 411899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 416      |
|    fps             | 34       |
|    time_elapsed    | 11952    |
|    total_timesteps | 416000   |
| train/             |          |
|    actor_loss      | -415     |
|    critic_loss     | 3.5      |
|    ent_coef        | 0.117    |
|    ent_coef_loss   | 1.3      |
|    learning_rate   | 0.0003   |
|    n_updates       | 415899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 420      |
|    fps             | 34       |
|    time_elapsed    | 12077    |
|    total_timesteps | 420000   |
| train/             |          |
|    actor_loss      | -408     |
|    critic_loss     | 3.46     |
|    ent_coef        | 0.119    |
|    ent_coef_loss   | -0.235   |
|    learning_rate   | 0.0003   |
|    n_updates       | 419899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 424      |
|    fps             | 34       |
|    time_elapsed    | 12200    |
|    total_timesteps | 424000   |
| train/             |          |
|    actor_loss      | -415     |
|    critic_loss     | 3.05     |
|    ent_coef        | 0.118    |
|    ent_coef_loss   | -0.472   |
|    learning_rate   | 0.0003   |
|    n_updates       | 423899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 428      |
|    fps             | 34       |
|    time_elapsed    | 12322    |
|    total_timesteps | 428000   |
| train/             |          |
|    actor_loss      | -408     |
|    critic_loss     | 4.84     |
|    ent_coef        | 0.119    |
|    ent_coef_loss   | 0.262    |
|    learning_rate   | 0.0003   |
|    n_updates       | 427899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 432      |
|    fps             | 34       |
|    time_elapsed    | 12438    |
|    total_timesteps | 432000   |
| train/             |          |
|    actor_loss      | -417     |
|    critic_loss     | 3.04     |
|    ent_coef        | 0.119    |
|    ent_coef_loss   | 0.332    |
|    learning_rate   | 0.0003   |
|    n_updates       | 431899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 436      |
|    fps             | 34       |
|    time_elapsed    | 12546    |
|    total_timesteps | 436000   |
| train/             |          |
|    actor_loss      | -392     |
|    critic_loss     | 2.28     |
|    ent_coef        | 0.118    |
|    ent_coef_loss   | -0.0425  |
|    learning_rate   | 0.0003   |
|    n_updates       | 435899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 440      |
|    fps             | 34       |
|    time_elapsed    | 12652    |
|    total_timesteps | 440000   |
| train/             |          |
|    actor_loss      | -415     |
|    critic_loss     | 3.82     |
|    ent_coef        | 0.119    |
|    ent_coef_loss   | 0.421    |
|    learning_rate   | 0.0003   |
|    n_updates       | 439899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 444      |
|    fps             | 34       |
|    time_elapsed    | 12759    |
|    total_timesteps | 444000   |
| train/             |          |
|    actor_loss      | -408     |
|    critic_loss     | 3.33     |
|    ent_coef        | 0.121    |
|    ent_coef_loss   | -1.1     |
|    learning_rate   | 0.0003   |
|    n_updates       | 443899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 448      |
|    fps             | 34       |
|    time_elapsed    | 12866    |
|    total_timesteps | 448000   |
| train/             |          |
|    actor_loss      | -414     |
|    critic_loss     | 3.81     |
|    ent_coef        | 0.119    |
|    ent_coef_loss   | 0.472    |
|    learning_rate   | 0.0003   |
|    n_updates       | 447899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 452      |
|    fps             | 34       |
|    time_elapsed    | 12973    |
|    total_timesteps | 452000   |
| train/             |          |
|    actor_loss      | -424     |
|    critic_loss     | 3.82     |
|    ent_coef        | 0.119    |
|    ent_coef_loss   | 0.0462   |
|    learning_rate   | 0.0003   |
|    n_updates       | 451899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 456      |
|    fps             | 34       |
|    time_elapsed    | 13067    |
|    total_timesteps | 456000   |
| train/             |          |
|    actor_loss      | -420     |
|    critic_loss     | 7.68     |
|    ent_coef        | 0.121    |
|    ent_coef_loss   | 0.421    |
|    learning_rate   | 0.0003   |
|    n_updates       | 455899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 460      |
|    fps             | 34       |
|    time_elapsed    | 13158    |
|    total_timesteps | 460000   |
| train/             |          |
|    actor_loss      | -416     |
|    critic_loss     | 4.01     |
|    ent_coef        | 0.12     |
|    ent_coef_loss   | 0.74     |
|    learning_rate   | 0.0003   |
|    n_updates       | 459899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 464      |
|    fps             | 34       |
|    time_elapsed    | 13259    |
|    total_timesteps | 464000   |
| train/             |          |
|    actor_loss      | -400     |
|    critic_loss     | 2.66     |
|    ent_coef        | 0.12     |
|    ent_coef_loss   | -0.151   |
|    learning_rate   | 0.0003   |
|    n_updates       | 463899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 468      |
|    fps             | 35       |
|    time_elapsed    | 13353    |
|    total_timesteps | 468000   |
| train/             |          |
|    actor_loss      | -429     |
|    critic_loss     | 3        |
|    ent_coef        | 0.118    |
|    ent_coef_loss   | -0.0239  |
|    learning_rate   | 0.0003   |
|    n_updates       | 467899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 472      |
|    fps             | 35       |
|    time_elapsed    | 13437    |
|    total_timesteps | 472000   |
| train/             |          |
|    actor_loss      | -427     |
|    critic_loss     | 4.01     |
|    ent_coef        | 0.118    |
|    ent_coef_loss   | 0.787    |
|    learning_rate   | 0.0003   |
|    n_updates       | 471899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 476      |
|    fps             | 35       |
|    time_elapsed    | 13494    |
|    total_timesteps | 476000   |
| train/             |          |
|    actor_loss      | -419     |
|    critic_loss     | 2.34     |
|    ent_coef        | 0.118    |
|    ent_coef_loss   | -0.463   |
|    learning_rate   | 0.0003   |
|    n_updates       | 475899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 480      |
|    fps             | 35       |
|    time_elapsed    | 13550    |
|    total_timesteps | 480000   |
| train/             |          |
|    actor_loss      | -418     |
|    critic_loss     | 2.58     |
|    ent_coef        | 0.121    |
|    ent_coef_loss   | -0.451   |
|    learning_rate   | 0.0003   |
|    n_updates       | 479899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 484      |
|    fps             | 35       |
|    time_elapsed    | 13594    |
|    total_timesteps | 484000   |
| train/             |          |
|    actor_loss      | -429     |
|    critic_loss     | 3.79     |
|    ent_coef        | 0.119    |
|    ent_coef_loss   | 0.31     |
|    learning_rate   | 0.0003   |
|    n_updates       | 483899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 488      |
|    fps             | 35       |
|    time_elapsed    | 13642    |
|    total_timesteps | 488000   |
| train/             |          |
|    actor_loss      | -431     |
|    critic_loss     | 4.52     |
|    ent_coef        | 0.12     |
|    ent_coef_loss   | 0.615    |
|    learning_rate   | 0.0003   |
|    n_updates       | 487899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 492      |
|    fps             | 35       |
|    time_elapsed    | 13688    |
|    total_timesteps | 492000   |
| train/             |          |
|    actor_loss      | -423     |
|    critic_loss     | 3.44     |
|    ent_coef        | 0.12     |
|    ent_coef_loss   | -1.09    |
|    learning_rate   | 0.0003   |
|    n_updates       | 491899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 496      |
|    fps             | 36       |
|    time_elapsed    | 13739    |
|    total_timesteps | 496000   |
| train/             |          |
|    actor_loss      | -421     |
|    critic_loss     | 2.9      |
|    ent_coef        | 0.12     |
|    ent_coef_loss   | -0.817   |
|    learning_rate   | 0.0003   |
|    n_updates       | 495899   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 500      |
|    fps             | 36       |
|    time_elapsed    | 13784    |
|    total_timesteps | 500000   |
| train/             |          |
|    actor_loss      | -425     |
|    critic_loss     | 3.82     |
|    ent_coef        | 0.12     |
|    ent_coef_loss   | -0.0104  |
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
/home/zhushaoq/SB_FVRL/utils/algorithmwrappers.py:1564: UserWarning: Using a target size (torch.Size([256, 256])) that is different to the input size (torch.Size([256, 1])). This will likely lead to incorrect results due to broadcasting. Please ensure they have the same size.
  critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
