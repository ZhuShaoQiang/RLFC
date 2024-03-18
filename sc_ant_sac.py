nohup: ignoring input
Using cuda device
SACPolicy(
  (actor): Actor(
    (features_extractor): FlattenExtractor(
      (flatten): Flatten(start_dim=1, end_dim=-1)
    )
    (latent_pi): Sequential(
      (0): Linear(in_features=27, out_features=256, bias=True)
      (1): ReLU()
      (2): Linear(in_features=256, out_features=256, bias=True)
      (3): ReLU()
    )
    (mu): Linear(in_features=256, out_features=8, bias=True)
    (log_std): Linear(in_features=256, out_features=8, bias=True)
  )
  (critic): ContinuousCritic(
    (features_extractor): FlattenExtractor(
      (flatten): Flatten(start_dim=1, end_dim=-1)
    )
    (qf0): Sequential(
      (0): Linear(in_features=35, out_features=256, bias=True)
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
      (0): Linear(in_features=35, out_features=256, bias=True)
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
|    fps             | 54       |
|    time_elapsed    | 4        |
|    total_timesteps | 226      |
| train/             |          |
|    actor_loss      | -11.9    |
|    critic_loss     | 1.25     |
|    ent_coef        | 0.963    |
|    ent_coef_loss   | -0.5     |
|    learning_rate   | 0.0003   |
|    n_updates       | 125      |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 8        |
|    fps             | 49       |
|    time_elapsed    | 8        |
|    total_timesteps | 425      |
| train/             |          |
|    actor_loss      | -14.7    |
|    critic_loss     | 1.22     |
|    ent_coef        | 0.908    |
|    ent_coef_loss   | -1.3     |
|    learning_rate   | 0.0003   |
|    n_updates       | 324      |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 12       |
|    fps             | 51       |
|    time_elapsed    | 17       |
|    total_timesteps | 908      |
| train/             |          |
|    actor_loss      | -21.5    |
|    critic_loss     | 0.79     |
|    ent_coef        | 0.786    |
|    ent_coef_loss   | -3.17    |
|    learning_rate   | 0.0003   |
|    n_updates       | 807      |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 16       |
|    fps             | 50       |
|    time_elapsed    | 39       |
|    total_timesteps | 1991     |
| train/             |          |
|    actor_loss      | -31.8    |
|    critic_loss     | 1.93     |
|    ent_coef        | 0.573    |
|    ent_coef_loss   | -6.87    |
|    learning_rate   | 0.0003   |
|    n_updates       | 1890     |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 20       |
|    fps             | 51       |
|    time_elapsed    | 44       |
|    total_timesteps | 2276     |
| train/             |          |
|    actor_loss      | -34.6    |
|    critic_loss     | 1.53     |
|    ent_coef        | 0.527    |
|    ent_coef_loss   | -7.82    |
|    learning_rate   | 0.0003   |
|    n_updates       | 2175     |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 24       |
|    fps             | 49       |
|    time_elapsed    | 51       |
|    total_timesteps | 2569     |
| train/             |          |
|    actor_loss      | -36.7    |
|    critic_loss     | 2.03     |
|    ent_coef        | 0.485    |
|    ent_coef_loss   | -8.55    |
|    learning_rate   | 0.0003   |
|    n_updates       | 2468     |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 28       |
|    fps             | 44       |
|    time_elapsed    | 82       |
|    total_timesteps | 3682     |
| train/             |          |
|    actor_loss      | -45      |
|    critic_loss     | 3.35     |
|    ent_coef        | 0.356    |
|    ent_coef_loss   | -11      |
|    learning_rate   | 0.0003   |
|    n_updates       | 3581     |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 32       |
|    fps             | 42       |
|    time_elapsed    | 116      |
|    total_timesteps | 4942     |
| train/             |          |
|    actor_loss      | -50.3    |
|    critic_loss     | 4.51     |
|    ent_coef        | 0.254    |
|    ent_coef_loss   | -11.7    |
|    learning_rate   | 0.0003   |
|    n_updates       | 4841     |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 36       |
|    fps             | 41       |
|    time_elapsed    | 144      |
|    total_timesteps | 6042     |
| train/             |          |
|    actor_loss      | -55.5    |
|    critic_loss     | 4.3      |
|    ent_coef        | 0.192    |
|    ent_coef_loss   | -12.6    |
|    learning_rate   | 0.0003   |
|    n_updates       | 5941     |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 40       |
|    fps             | 40       |
|    time_elapsed    | 198      |
|    total_timesteps | 8083     |
| train/             |          |
|    actor_loss      | -65.2    |
|    critic_loss     | 4.97     |
|    ent_coef        | 0.116    |
|    ent_coef_loss   | -10.5    |
|    learning_rate   | 0.0003   |
|    n_updates       | 7982     |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 44       |
|    fps             | 40       |
|    time_elapsed    | 201      |
|    total_timesteps | 8209     |
| train/             |          |
|    actor_loss      | -66.1    |
|    critic_loss     | 5.04     |
|    ent_coef        | 0.113    |
|    ent_coef_loss   | -10.1    |
|    learning_rate   | 0.0003   |
|    n_updates       | 8108     |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 48       |
|    fps             | 40       |
|    time_elapsed    | 208      |
|    total_timesteps | 8499     |
| train/             |          |
|    actor_loss      | -66.3    |
|    critic_loss     | 15.1     |
|    ent_coef        | 0.105    |
|    ent_coef_loss   | -9.67    |
|    learning_rate   | 0.0003   |
|    n_updates       | 8398     |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 52       |
|    fps             | 40       |
|    time_elapsed    | 215      |
|    total_timesteps | 8752     |
| train/             |          |
|    actor_loss      | -66.7    |
|    critic_loss     | 6.8      |
|    ent_coef        | 0.0995   |
|    ent_coef_loss   | -9.17    |
|    learning_rate   | 0.0003   |
|    n_updates       | 8651     |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 56       |
|    fps             | 40       |
|    time_elapsed    | 245      |
|    total_timesteps | 9908     |
| train/             |          |
|    actor_loss      | -69.5    |
|    critic_loss     | 7.77     |
|    ent_coef        | 0.0774   |
|    ent_coef_loss   | -7.91    |
|    learning_rate   | 0.0003   |
|    n_updates       | 9807     |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 60       |
|    fps             | 39       |
|    time_elapsed    | 302      |
|    total_timesteps | 11995    |
| train/             |          |
|    actor_loss      | -75.8    |
|    critic_loss     | 7.28     |
|    ent_coef        | 0.0545   |
|    ent_coef_loss   | -2.2     |
|    learning_rate   | 0.0003   |
|    n_updates       | 11894    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 64       |
|    fps             | 39       |
|    time_elapsed    | 304      |
|    total_timesteps | 12096    |
| train/             |          |
|    actor_loss      | -75.8    |
|    critic_loss     | 6.71     |
|    ent_coef        | 0.0537   |
|    ent_coef_loss   | -3.91    |
|    learning_rate   | 0.0003   |
|    n_updates       | 11995    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 68       |
|    fps             | 39       |
|    time_elapsed    | 308      |
|    total_timesteps | 12227    |
| train/             |          |
|    actor_loss      | -75.7    |
|    critic_loss     | 5.75     |
|    ent_coef        | 0.0529   |
|    ent_coef_loss   | -2.87    |
|    learning_rate   | 0.0003   |
|    n_updates       | 12126    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 72       |
|    fps             | 39       |
|    time_elapsed    | 366      |
|    total_timesteps | 14423    |
| train/             |          |
|    actor_loss      | -78.8    |
|    critic_loss     | 7.34     |
|    ent_coef        | 0.0432   |
|    ent_coef_loss   | -0.871   |
|    learning_rate   | 0.0003   |
|    n_updates       | 14322    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 76       |
|    fps             | 39       |
|    time_elapsed    | 375      |
|    total_timesteps | 14717    |
| train/             |          |
|    actor_loss      | -81.6    |
|    critic_loss     | 6.25     |
|    ent_coef        | 0.0425   |
|    ent_coef_loss   | 0.284    |
|    learning_rate   | 0.0003   |
|    n_updates       | 14616    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 80       |
|    fps             | 39       |
|    time_elapsed    | 405      |
|    total_timesteps | 15814    |
| train/             |          |
|    actor_loss      | -82.2    |
|    critic_loss     | 7.67     |
|    ent_coef        | 0.0402   |
|    ent_coef_loss   | -0.136   |
|    learning_rate   | 0.0003   |
|    n_updates       | 15713    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 84       |
|    fps             | 38       |
|    time_elapsed    | 462      |
|    total_timesteps | 17913    |
| train/             |          |
|    actor_loss      | -83.1    |
|    critic_loss     | 10       |
|    ent_coef        | 0.0363   |
|    ent_coef_loss   | -0.0245  |
|    learning_rate   | 0.0003   |
|    n_updates       | 17812    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 88       |
|    fps             | 38       |
|    time_elapsed    | 468      |
|    total_timesteps | 18140    |
| train/             |          |
|    actor_loss      | -83.4    |
|    critic_loss     | 6.85     |
|    ent_coef        | 0.0361   |
|    ent_coef_loss   | -1.08    |
|    learning_rate   | 0.0003   |
|    n_updates       | 18039    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 92       |
|    fps             | 38       |
|    time_elapsed    | 476      |
|    total_timesteps | 18486    |
| train/             |          |
|    actor_loss      | -84.2    |
|    critic_loss     | 7.51     |
|    ent_coef        | 0.0352   |
|    ent_coef_loss   | -1.36    |
|    learning_rate   | 0.0003   |
|    n_updates       | 18385    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 96       |
|    fps             | 38       |
|    time_elapsed    | 542      |
|    total_timesteps | 20761    |
| train/             |          |
|    actor_loss      | -83.9    |
|    critic_loss     | 6.58     |
|    ent_coef        | 0.0326   |
|    ent_coef_loss   | -0.3     |
|    learning_rate   | 0.0003   |
|    n_updates       | 20660    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 100      |
|    fps             | 38       |
|    time_elapsed    | 549      |
|    total_timesteps | 20979    |
| train/             |          |
|    actor_loss      | -86.3    |
|    critic_loss     | 7.49     |
|    ent_coef        | 0.0326   |
|    ent_coef_loss   | -0.937   |
|    learning_rate   | 0.0003   |
|    n_updates       | 20878    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 104      |
|    fps             | 38       |
|    time_elapsed    | 579      |
|    total_timesteps | 22103    |
| train/             |          |
|    actor_loss      | -85      |
|    critic_loss     | 7.09     |
|    ent_coef        | 0.0315   |
|    ent_coef_loss   | 1.46     |
|    learning_rate   | 0.0003   |
|    n_updates       | 22002    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 108      |
|    fps             | 38       |
|    time_elapsed    | 590      |
|    total_timesteps | 22473    |
| train/             |          |
|    actor_loss      | -86.1    |
|    critic_loss     | 7.51     |
|    ent_coef        | 0.0307   |
|    ent_coef_loss   | -2.29    |
|    learning_rate   | 0.0003   |
|    n_updates       | 22372    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 112      |
|    fps             | 37       |
|    time_elapsed    | 674      |
|    total_timesteps | 25536    |
| train/             |          |
|    actor_loss      | -84.1    |
|    critic_loss     | 4.82     |
|    ent_coef        | 0.0272   |
|    ent_coef_loss   | -0.301   |
|    learning_rate   | 0.0003   |
|    n_updates       | 25435    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 116      |
|    fps             | 37       |
|    time_elapsed    | 734      |
|    total_timesteps | 27702    |
| train/             |          |
|    actor_loss      | -81.9    |
|    critic_loss     | 4.32     |
|    ent_coef        | 0.0245   |
|    ent_coef_loss   | 1.97     |
|    learning_rate   | 0.0003   |
|    n_updates       | 27601    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 120      |
|    fps             | 37       |
|    time_elapsed    | 791      |
|    total_timesteps | 29852    |
| train/             |          |
|    actor_loss      | -79.8    |
|    critic_loss     | 4.09     |
|    ent_coef        | 0.0212   |
|    ent_coef_loss   | 0.646    |
|    learning_rate   | 0.0003   |
|    n_updates       | 29751    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 124      |
|    fps             | 37       |
|    time_elapsed    | 826      |
|    total_timesteps | 31076    |
| train/             |          |
|    actor_loss      | -77.1    |
|    critic_loss     | 5.09     |
|    ent_coef        | 0.0201   |
|    ent_coef_loss   | -0.368   |
|    learning_rate   | 0.0003   |
|    n_updates       | 30975    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 128      |
|    fps             | 37       |
|    time_elapsed    | 858      |
|    total_timesteps | 32294    |
| train/             |          |
|    actor_loss      | -76.1    |
|    critic_loss     | 3.19     |
|    ent_coef        | 0.0186   |
|    ent_coef_loss   | -0.727   |
|    learning_rate   | 0.0003   |
|    n_updates       | 32193    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 132      |
|    fps             | 37       |
|    time_elapsed    | 893      |
|    total_timesteps | 33529    |
| train/             |          |
|    actor_loss      | -74.7    |
|    critic_loss     | 3.74     |
|    ent_coef        | 0.0172   |
|    ent_coef_loss   | -0.898   |
|    learning_rate   | 0.0003   |
|    n_updates       | 33428    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 136      |
|    fps             | 37       |
|    time_elapsed    | 975      |
|    total_timesteps | 36653    |
| train/             |          |
|    actor_loss      | -69.1    |
|    critic_loss     | 3.05     |
|    ent_coef        | 0.0147   |
|    ent_coef_loss   | -0.393   |
|    learning_rate   | 0.0003   |
|    n_updates       | 36552    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 140      |
|    fps             | 37       |
|    time_elapsed    | 1045     |
|    total_timesteps | 39237    |
| train/             |          |
|    actor_loss      | -65      |
|    critic_loss     | 3.44     |
|    ent_coef        | 0.0132   |
|    ent_coef_loss   | -0.59    |
|    learning_rate   | 0.0003   |
|    n_updates       | 39136    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 144      |
|    fps             | 37       |
|    time_elapsed    | 1081     |
|    total_timesteps | 40576    |
| train/             |          |
|    actor_loss      | -65.1    |
|    critic_loss     | 2.5      |
|    ent_coef        | 0.0126   |
|    ent_coef_loss   | 0.69     |
|    learning_rate   | 0.0003   |
|    n_updates       | 40475    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 148      |
|    fps             | 37       |
|    time_elapsed    | 1092     |
|    total_timesteps | 40954    |
| train/             |          |
|    actor_loss      | -65      |
|    critic_loss     | 3.4      |
|    ent_coef        | 0.0123   |
|    ent_coef_loss   | -0.0954  |
|    learning_rate   | 0.0003   |
|    n_updates       | 40853    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 152      |
|    fps             | 37       |
|    time_elapsed    | 1147     |
|    total_timesteps | 43014    |
| train/             |          |
|    actor_loss      | -60.8    |
|    critic_loss     | 2.55     |
|    ent_coef        | 0.0117   |
|    ent_coef_loss   | -0.262   |
|    learning_rate   | 0.0003   |
|    n_updates       | 42913    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 156      |
|    fps             | 37       |
|    time_elapsed    | 1166     |
|    total_timesteps | 43739    |
| train/             |          |
|    actor_loss      | -61.6    |
|    critic_loss     | 3.03     |
|    ent_coef        | 0.0116   |
|    ent_coef_loss   | 0.156    |
|    learning_rate   | 0.0003   |
|    n_updates       | 43638    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 160      |
|    fps             | 37       |
|    time_elapsed    | 1177     |
|    total_timesteps | 44153    |
| train/             |          |
|    actor_loss      | -61.5    |
|    critic_loss     | 2.32     |
|    ent_coef        | 0.0115   |
|    ent_coef_loss   | -0.195   |
|    learning_rate   | 0.0003   |
|    n_updates       | 44052    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 164      |
|    fps             | 37       |
|    time_elapsed    | 1215     |
|    total_timesteps | 45445    |
| train/             |          |
|    actor_loss      | -60.4    |
|    critic_loss     | 2.26     |
|    ent_coef        | 0.0112   |
|    ent_coef_loss   | 0.544    |
|    learning_rate   | 0.0003   |
|    n_updates       | 45344    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 168      |
|    fps             | 37       |
|    time_elapsed    | 1233     |
|    total_timesteps | 46142    |
| train/             |          |
|    actor_loss      | -58.2    |
|    critic_loss     | 2.8      |
|    ent_coef        | 0.0111   |
|    ent_coef_loss   | 0.593    |
|    learning_rate   | 0.0003   |
|    n_updates       | 46041    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 172      |
|    fps             | 37       |
|    time_elapsed    | 1291     |
|    total_timesteps | 48353    |
| train/             |          |
|    actor_loss      | -58.1    |
|    critic_loss     | 2.72     |
|    ent_coef        | 0.0106   |
|    ent_coef_loss   | 0.426    |
|    learning_rate   | 0.0003   |
|    n_updates       | 48252    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 176      |
|    fps             | 37       |
|    time_elapsed    | 1306     |
|    total_timesteps | 48904    |
| train/             |          |
|    actor_loss      | -58.7    |
|    critic_loss     | 3.4      |
|    ent_coef        | 0.0106   |
|    ent_coef_loss   | 1.12     |
|    learning_rate   | 0.0003   |
|    n_updates       | 48803    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 180      |
|    fps             | 37       |
|    time_elapsed    | 1320     |
|    total_timesteps | 49390    |
| train/             |          |
|    actor_loss      | -56.3    |
|    critic_loss     | 1.9      |
|    ent_coef        | 0.0104   |
|    ent_coef_loss   | -2.22    |
|    learning_rate   | 0.0003   |
|    n_updates       | 49289    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 184      |
|    fps             | 37       |
|    time_elapsed    | 1330     |
|    total_timesteps | 49763    |
| train/             |          |
|    actor_loss      | -56.3    |
|    critic_loss     | 2.51     |
|    ent_coef        | 0.0104   |
|    ent_coef_loss   | -0.943   |
|    learning_rate   | 0.0003   |
|    n_updates       | 49662    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 188      |
|    fps             | 37       |
|    time_elapsed    | 1412     |
|    total_timesteps | 52789    |
| train/             |          |
|    actor_loss      | -54.8    |
|    critic_loss     | 2.07     |
|    ent_coef        | 0.01     |
|    ent_coef_loss   | -0.08    |
|    learning_rate   | 0.0003   |
|    n_updates       | 52688    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 192      |
|    fps             | 37       |
|    time_elapsed    | 1443     |
|    total_timesteps | 53936    |
| train/             |          |
|    actor_loss      | -54.5    |
|    critic_loss     | 2.45     |
|    ent_coef        | 0.0102   |
|    ent_coef_loss   | 1.21     |
|    learning_rate   | 0.0003   |
|    n_updates       | 53835    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 196      |
|    fps             | 37       |
|    time_elapsed    | 1457     |
|    total_timesteps | 54435    |
| train/             |          |
|    actor_loss      | -54      |
|    critic_loss     | 2.52     |
|    ent_coef        | 0.00977  |
|    ent_coef_loss   | 0.014    |
|    learning_rate   | 0.0003   |
|    n_updates       | 54334    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 200      |
|    fps             | 37       |
|    time_elapsed    | 1462     |
|    total_timesteps | 54612    |
| train/             |          |
|    actor_loss      | -55.9    |
|    critic_loss     | 2.1      |
|    ent_coef        | 0.00979  |
|    ent_coef_loss   | -1.27    |
|    learning_rate   | 0.0003   |
|    n_updates       | 54511    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 204      |
|    fps             | 37       |
|    time_elapsed    | 1507     |
|    total_timesteps | 56235    |
| train/             |          |
|    actor_loss      | -53      |
|    critic_loss     | 1.71     |
|    ent_coef        | 0.00968  |
|    ent_coef_loss   | 0.416    |
|    learning_rate   | 0.0003   |
|    n_updates       | 56134    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 208      |
|    fps             | 37       |
|    time_elapsed    | 1521     |
|    total_timesteps | 56765    |
| train/             |          |
|    actor_loss      | -54      |
|    critic_loss     | 2.36     |
|    ent_coef        | 0.00971  |
|    ent_coef_loss   | 0.321    |
|    learning_rate   | 0.0003   |
|    n_updates       | 56664    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 212      |
|    fps             | 37       |
|    time_elapsed    | 1565     |
|    total_timesteps | 58369    |
| train/             |          |
|    actor_loss      | -52.9    |
|    critic_loss     | 2.21     |
|    ent_coef        | 0.0096   |
|    ent_coef_loss   | -2.16    |
|    learning_rate   | 0.0003   |
|    n_updates       | 58268    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 216      |
|    fps             | 37       |
|    time_elapsed    | 1629     |
|    total_timesteps | 60677    |
| train/             |          |
|    actor_loss      | -54.7    |
|    critic_loss     | 3.14     |
|    ent_coef        | 0.00932  |
|    ent_coef_loss   | -0.0104  |
|    learning_rate   | 0.0003   |
|    n_updates       | 60576    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 220      |
|    fps             | 37       |
|    time_elapsed    | 1659     |
|    total_timesteps | 61822    |
| train/             |          |
|    actor_loss      | -53.3    |
|    critic_loss     | 2.01     |
|    ent_coef        | 0.00924  |
|    ent_coef_loss   | -0.283   |
|    learning_rate   | 0.0003   |
|    n_updates       | 61721    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 224      |
|    fps             | 37       |
|    time_elapsed    | 1685     |
|    total_timesteps | 62811    |
| train/             |          |
|    actor_loss      | -51.8    |
|    critic_loss     | 2.3      |
|    ent_coef        | 0.00917  |
|    ent_coef_loss   | -0.115   |
|    learning_rate   | 0.0003   |
|    n_updates       | 62710    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 228      |
|    fps             | 37       |
|    time_elapsed    | 1740     |
|    total_timesteps | 64896    |
| train/             |          |
|    actor_loss      | -50.7    |
|    critic_loss     | 2.16     |
|    ent_coef        | 0.00917  |
|    ent_coef_loss   | 0.657    |
|    learning_rate   | 0.0003   |
|    n_updates       | 64795    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 232      |
|    fps             | 37       |
|    time_elapsed    | 1793     |
|    total_timesteps | 66923    |
| train/             |          |
|    actor_loss      | -53.8    |
|    critic_loss     | 2.4      |
|    ent_coef        | 0.00892  |
|    ent_coef_loss   | -0.948   |
|    learning_rate   | 0.0003   |
|    n_updates       | 66822    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 236      |
|    fps             | 37       |
|    time_elapsed    | 1804     |
|    total_timesteps | 67357    |
| train/             |          |
|    actor_loss      | -52.2    |
|    critic_loss     | 1.81     |
|    ent_coef        | 0.00909  |
|    ent_coef_loss   | 0.15     |
|    learning_rate   | 0.0003   |
|    n_updates       | 67256    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 240      |
|    fps             | 37       |
|    time_elapsed    | 1853     |
|    total_timesteps | 69488    |
| train/             |          |
|    actor_loss      | -53.1    |
|    critic_loss     | 1.84     |
|    ent_coef        | 0.00906  |
|    ent_coef_loss   | 0.374    |
|    learning_rate   | 0.0003   |
|    n_updates       | 69387    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 244      |
|    fps             | 37       |
|    time_elapsed    | 1887     |
|    total_timesteps | 70728    |
| train/             |          |
|    actor_loss      | -53      |
|    critic_loss     | 1.82     |
|    ent_coef        | 0.00904  |
|    ent_coef_loss   | 0.0717   |
|    learning_rate   | 0.0003   |
|    n_updates       | 70627    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 248      |
|    fps             | 37       |
|    time_elapsed    | 1928     |
|    total_timesteps | 72405    |
| train/             |          |
|    actor_loss      | -55.3    |
|    critic_loss     | 2.12     |
|    ent_coef        | 0.009    |
|    ent_coef_loss   | 0.916    |
|    learning_rate   | 0.0003   |
|    n_updates       | 72304    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 252      |
|    fps             | 37       |
|    time_elapsed    | 1962     |
|    total_timesteps | 73818    |
| train/             |          |
|    actor_loss      | -53.8    |
|    critic_loss     | 1.76     |
|    ent_coef        | 0.00899  |
|    ent_coef_loss   | -0.522   |
|    learning_rate   | 0.0003   |
|    n_updates       | 73717    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 256      |
|    fps             | 37       |
|    time_elapsed    | 2012     |
|    total_timesteps | 75908    |
| train/             |          |
|    actor_loss      | -56      |
|    critic_loss     | 2.33     |
|    ent_coef        | 0.00887  |
|    ent_coef_loss   | -0.104   |
|    learning_rate   | 0.0003   |
|    n_updates       | 75807    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 260      |
|    fps             | 37       |
|    time_elapsed    | 2046     |
|    total_timesteps | 77163    |
| train/             |          |
|    actor_loss      | -54.9    |
|    critic_loss     | 1.59     |
|    ent_coef        | 0.00905  |
|    ent_coef_loss   | -3.09    |
|    learning_rate   | 0.0003   |
|    n_updates       | 77062    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 264      |
|    fps             | 37       |
|    time_elapsed    | 2076     |
|    total_timesteps | 78314    |
| train/             |          |
|    actor_loss      | -56.9    |
|    critic_loss     | 2.12     |
|    ent_coef        | 0.00909  |
|    ent_coef_loss   | 2.53     |
|    learning_rate   | 0.0003   |
|    n_updates       | 78213    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 268      |
|    fps             | 37       |
|    time_elapsed    | 2081     |
|    total_timesteps | 78514    |
| train/             |          |
|    actor_loss      | -55.7    |
|    critic_loss     | 1.57     |
|    ent_coef        | 0.0092   |
|    ent_coef_loss   | 1.97     |
|    learning_rate   | 0.0003   |
|    n_updates       | 78413    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 272      |
|    fps             | 37       |
|    time_elapsed    | 2138     |
|    total_timesteps | 80758    |
| train/             |          |
|    actor_loss      | -58.9    |
|    critic_loss     | 2.84     |
|    ent_coef        | 0.00932  |
|    ent_coef_loss   | -0.681   |
|    learning_rate   | 0.0003   |
|    n_updates       | 80657    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 276      |
|    fps             | 37       |
|    time_elapsed    | 2194     |
|    total_timesteps | 82826    |
| train/             |          |
|    actor_loss      | -56.2    |
|    critic_loss     | 1.79     |
|    ent_coef        | 0.00912  |
|    ent_coef_loss   | -1.32    |
|    learning_rate   | 0.0003   |
|    n_updates       | 82725    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 280      |
|    fps             | 37       |
|    time_elapsed    | 2238     |
|    total_timesteps | 84478    |
| train/             |          |
|    actor_loss      | -58.1    |
|    critic_loss     | 2.1      |
|    ent_coef        | 0.00924  |
|    ent_coef_loss   | -0.523   |
|    learning_rate   | 0.0003   |
|    n_updates       | 84377    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 284      |
|    fps             | 37       |
|    time_elapsed    | 2303     |
|    total_timesteps | 86720    |
| train/             |          |
|    actor_loss      | -57.9    |
|    critic_loss     | 1.93     |
|    ent_coef        | 0.00907  |
|    ent_coef_loss   | -0.379   |
|    learning_rate   | 0.0003   |
|    n_updates       | 86619    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 288      |
|    fps             | 37       |
|    time_elapsed    | 2384     |
|    total_timesteps | 89893    |
| train/             |          |
|    actor_loss      | -60.7    |
|    critic_loss     | 1.66     |
|    ent_coef        | 0.00929  |
|    ent_coef_loss   | 0.477    |
|    learning_rate   | 0.0003   |
|    n_updates       | 89792    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 292      |
|    fps             | 37       |
|    time_elapsed    | 2420     |
|    total_timesteps | 91208    |
| train/             |          |
|    actor_loss      | -60.6    |
|    critic_loss     | 1.57     |
|    ent_coef        | 0.00932  |
|    ent_coef_loss   | 0.706    |
|    learning_rate   | 0.0003   |
|    n_updates       | 91107    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 296      |
|    fps             | 37       |
|    time_elapsed    | 2455     |
|    total_timesteps | 92623    |
| train/             |          |
|    actor_loss      | -61      |
|    critic_loss     | 1.47     |
|    ent_coef        | 0.00939  |
|    ent_coef_loss   | -0.153   |
|    learning_rate   | 0.0003   |
|    n_updates       | 92522    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 300      |
|    fps             | 37       |
|    time_elapsed    | 2492     |
|    total_timesteps | 94116    |
| train/             |          |
|    actor_loss      | -65.1    |
|    critic_loss     | 2.28     |
|    ent_coef        | 0.00952  |
|    ent_coef_loss   | 0.607    |
|    learning_rate   | 0.0003   |
|    n_updates       | 94015    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 304      |
|    fps             | 37       |
|    time_elapsed    | 2525     |
|    total_timesteps | 95341    |
| train/             |          |
|    actor_loss      | -67.2    |
|    critic_loss     | 2.23     |
|    ent_coef        | 0.00971  |
|    ent_coef_loss   | -1.41    |
|    learning_rate   | 0.0003   |
|    n_updates       | 95240    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 308      |
|    fps             | 37       |
|    time_elapsed    | 2625     |
|    total_timesteps | 99341    |
| train/             |          |
|    actor_loss      | -65      |
|    critic_loss     | 2.1      |
|    ent_coef        | 0.00976  |
|    ent_coef_loss   | 1.2      |
|    learning_rate   | 0.0003   |
|    n_updates       | 99240    |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 312      |
|    fps             | 37       |
|    time_elapsed    | 2681     |
|    total_timesteps | 101635   |
| train/             |          |
|    actor_loss      | -65.4    |
|    critic_loss     | 1.82     |
|    ent_coef        | 0.00961  |
|    ent_coef_loss   | 1.15     |
|    learning_rate   | 0.0003   |
|    n_updates       | 101534   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 316      |
|    fps             | 37       |
|    time_elapsed    | 2716     |
|    total_timesteps | 103002   |
| train/             |          |
|    actor_loss      | -67.4    |
|    critic_loss     | 2.06     |
|    ent_coef        | 0.00969  |
|    ent_coef_loss   | 0.302    |
|    learning_rate   | 0.0003   |
|    n_updates       | 102901   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 320      |
|    fps             | 37       |
|    time_elapsed    | 2755     |
|    total_timesteps | 104558   |
| train/             |          |
|    actor_loss      | -67.1    |
|    critic_loss     | 1.85     |
|    ent_coef        | 0.00985  |
|    ent_coef_loss   | 0.044    |
|    learning_rate   | 0.0003   |
|    n_updates       | 104457   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 324      |
|    fps             | 37       |
|    time_elapsed    | 2808     |
|    total_timesteps | 106637   |
| train/             |          |
|    actor_loss      | -68.3    |
|    critic_loss     | 2.33     |
|    ent_coef        | 0.00981  |
|    ent_coef_loss   | -0.669   |
|    learning_rate   | 0.0003   |
|    n_updates       | 106536   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 328      |
|    fps             | 37       |
|    time_elapsed    | 2873     |
|    total_timesteps | 109116   |
| train/             |          |
|    actor_loss      | -71.9    |
|    critic_loss     | 1.87     |
|    ent_coef        | 0.00983  |
|    ent_coef_loss   | -1.05    |
|    learning_rate   | 0.0003   |
|    n_updates       | 109015   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 332      |
|    fps             | 37       |
|    time_elapsed    | 2932     |
|    total_timesteps | 111359   |
| train/             |          |
|    actor_loss      | -75.5    |
|    critic_loss     | 2.28     |
|    ent_coef        | 0.00964  |
|    ent_coef_loss   | 1.17     |
|    learning_rate   | 0.0003   |
|    n_updates       | 111258   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 336      |
|    fps             | 38       |
|    time_elapsed    | 2963     |
|    total_timesteps | 112636   |
| train/             |          |
|    actor_loss      | -73.1    |
|    critic_loss     | 1.79     |
|    ent_coef        | 0.00971  |
|    ent_coef_loss   | 1.17     |
|    learning_rate   | 0.0003   |
|    n_updates       | 112535   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 340      |
|    fps             | 38       |
|    time_elapsed    | 3016     |
|    total_timesteps | 114797   |
| train/             |          |
|    actor_loss      | -73.8    |
|    critic_loss     | 1.96     |
|    ent_coef        | 0.00972  |
|    ent_coef_loss   | -1.42    |
|    learning_rate   | 0.0003   |
|    n_updates       | 114696   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 344      |
|    fps             | 38       |
|    time_elapsed    | 3078     |
|    total_timesteps | 117136   |
| train/             |          |
|    actor_loss      | -77.8    |
|    critic_loss     | 2.53     |
|    ent_coef        | 0.00959  |
|    ent_coef_loss   | 2.88     |
|    learning_rate   | 0.0003   |
|    n_updates       | 117035   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 348      |
|    fps             | 38       |
|    time_elapsed    | 3114     |
|    total_timesteps | 118538   |
| train/             |          |
|    actor_loss      | -79.2    |
|    critic_loss     | 3.6      |
|    ent_coef        | 0.00965  |
|    ent_coef_loss   | 1.42     |
|    learning_rate   | 0.0003   |
|    n_updates       | 118437   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 352      |
|    fps             | 38       |
|    time_elapsed    | 3165     |
|    total_timesteps | 120686   |
| train/             |          |
|    actor_loss      | -78.7    |
|    critic_loss     | 2.53     |
|    ent_coef        | 0.0097   |
|    ent_coef_loss   | -0.14    |
|    learning_rate   | 0.0003   |
|    n_updates       | 120585   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 356      |
|    fps             | 38       |
|    time_elapsed    | 3199     |
|    total_timesteps | 122143   |
| train/             |          |
|    actor_loss      | -79.1    |
|    critic_loss     | 1.84     |
|    ent_coef        | 0.00974  |
|    ent_coef_loss   | 0.13     |
|    learning_rate   | 0.0003   |
|    n_updates       | 122042   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 360      |
|    fps             | 38       |
|    time_elapsed    | 3244     |
|    total_timesteps | 123867   |
| train/             |          |
|    actor_loss      | -76.7    |
|    critic_loss     | 1.72     |
|    ent_coef        | 0.00973  |
|    ent_coef_loss   | -0.707   |
|    learning_rate   | 0.0003   |
|    n_updates       | 123766   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 364      |
|    fps             | 38       |
|    time_elapsed    | 3297     |
|    total_timesteps | 125977   |
| train/             |          |
|    actor_loss      | -78.1    |
|    critic_loss     | 1.72     |
|    ent_coef        | 0.00976  |
|    ent_coef_loss   | -0.351   |
|    learning_rate   | 0.0003   |
|    n_updates       | 125876   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 368      |
|    fps             | 38       |
|    time_elapsed    | 3375     |
|    total_timesteps | 129051   |
| train/             |          |
|    actor_loss      | -81.7    |
|    critic_loss     | 2.74     |
|    ent_coef        | 0.00975  |
|    ent_coef_loss   | 0.139    |
|    learning_rate   | 0.0003   |
|    n_updates       | 128950   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 372      |
|    fps             | 38       |
|    time_elapsed    | 3424     |
|    total_timesteps | 131104   |
| train/             |          |
|    actor_loss      | -81.3    |
|    critic_loss     | 3.26     |
|    ent_coef        | 0.00957  |
|    ent_coef_loss   | 0.897    |
|    learning_rate   | 0.0003   |
|    n_updates       | 131003   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 376      |
|    fps             | 38       |
|    time_elapsed    | 3487     |
|    total_timesteps | 133594   |
| train/             |          |
|    actor_loss      | -80.6    |
|    critic_loss     | 1.54     |
|    ent_coef        | 0.00995  |
|    ent_coef_loss   | -0.522   |
|    learning_rate   | 0.0003   |
|    n_updates       | 133493   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 380      |
|    fps             | 38       |
|    time_elapsed    | 3518     |
|    total_timesteps | 134912   |
| train/             |          |
|    actor_loss      | -76.8    |
|    critic_loss     | 2.46     |
|    ent_coef        | 0.00975  |
|    ent_coef_loss   | -0.324   |
|    learning_rate   | 0.0003   |
|    n_updates       | 134811   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 384      |
|    fps             | 38       |
|    time_elapsed    | 3549     |
|    total_timesteps | 136004   |
| train/             |          |
|    actor_loss      | -79.3    |
|    critic_loss     | 2.04     |
|    ent_coef        | 0.00978  |
|    ent_coef_loss   | 1.05     |
|    learning_rate   | 0.0003   |
|    n_updates       | 135903   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 388      |
|    fps             | 38       |
|    time_elapsed    | 3583     |
|    total_timesteps | 137385   |
| train/             |          |
|    actor_loss      | -74.7    |
|    critic_loss     | 1.56     |
|    ent_coef        | 0.00999  |
|    ent_coef_loss   | -0.408   |
|    learning_rate   | 0.0003   |
|    n_updates       | 137284   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 392      |
|    fps             | 38       |
|    time_elapsed    | 3642     |
|    total_timesteps | 139767   |
| train/             |          |
|    actor_loss      | -79.2    |
|    critic_loss     | 2.52     |
|    ent_coef        | 0.0101   |
|    ent_coef_loss   | 0.399    |
|    learning_rate   | 0.0003   |
|    n_updates       | 139666   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 396      |
|    fps             | 38       |
|    time_elapsed    | 3667     |
|    total_timesteps | 140722   |
| train/             |          |
|    actor_loss      | -82.7    |
|    critic_loss     | 1.66     |
|    ent_coef        | 0.0101   |
|    ent_coef_loss   | -0.643   |
|    learning_rate   | 0.0003   |
|    n_updates       | 140621   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 400      |
|    fps             | 38       |
|    time_elapsed    | 3722     |
|    total_timesteps | 142921   |
| train/             |          |
|    actor_loss      | -82.3    |
|    critic_loss     | 5.69     |
|    ent_coef        | 0.0103   |
|    ent_coef_loss   | 0.187    |
|    learning_rate   | 0.0003   |
|    n_updates       | 142820   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 404      |
|    fps             | 38       |
|    time_elapsed    | 3773     |
|    total_timesteps | 144969   |
| train/             |          |
|    actor_loss      | -84.3    |
|    critic_loss     | 5        |
|    ent_coef        | 0.0103   |
|    ent_coef_loss   | -0.28    |
|    learning_rate   | 0.0003   |
|    n_updates       | 144868   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 408      |
|    fps             | 38       |
|    time_elapsed    | 3835     |
|    total_timesteps | 147359   |
| train/             |          |
|    actor_loss      | -88.2    |
|    critic_loss     | 2.69     |
|    ent_coef        | 0.0102   |
|    ent_coef_loss   | -0.689   |
|    learning_rate   | 0.0003   |
|    n_updates       | 147258   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 412      |
|    fps             | 38       |
|    time_elapsed    | 3867     |
|    total_timesteps | 148623   |
| train/             |          |
|    actor_loss      | -84.3    |
|    critic_loss     | 5.84     |
|    ent_coef        | 0.0102   |
|    ent_coef_loss   | -1.62    |
|    learning_rate   | 0.0003   |
|    n_updates       | 148522   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 416      |
|    fps             | 38       |
|    time_elapsed    | 3900     |
|    total_timesteps | 149880   |
| train/             |          |
|    actor_loss      | -85.1    |
|    critic_loss     | 1.71     |
|    ent_coef        | 0.0103   |
|    ent_coef_loss   | -0.134   |
|    learning_rate   | 0.0003   |
|    n_updates       | 149779   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 420      |
|    fps             | 38       |
|    time_elapsed    | 3955     |
|    total_timesteps | 152102   |
| train/             |          |
|    actor_loss      | -86.1    |
|    critic_loss     | 2.67     |
|    ent_coef        | 0.0102   |
|    ent_coef_loss   | 2.32     |
|    learning_rate   | 0.0003   |
|    n_updates       | 152001   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 424      |
|    fps             | 38       |
|    time_elapsed    | 4012     |
|    total_timesteps | 154373   |
| train/             |          |
|    actor_loss      | -88.6    |
|    critic_loss     | 3.83     |
|    ent_coef        | 0.0101   |
|    ent_coef_loss   | -0.533   |
|    learning_rate   | 0.0003   |
|    n_updates       | 154272   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 428      |
|    fps             | 38       |
|    time_elapsed    | 4062     |
|    total_timesteps | 156379   |
| train/             |          |
|    actor_loss      | -88.6    |
|    critic_loss     | 4.09     |
|    ent_coef        | 0.0102   |
|    ent_coef_loss   | -3.27    |
|    learning_rate   | 0.0003   |
|    n_updates       | 156278   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 432      |
|    fps             | 38       |
|    time_elapsed    | 4118     |
|    total_timesteps | 158587   |
| train/             |          |
|    actor_loss      | -90      |
|    critic_loss     | 2.08     |
|    ent_coef        | 0.0101   |
|    ent_coef_loss   | 0.167    |
|    learning_rate   | 0.0003   |
|    n_updates       | 158486   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 436      |
|    fps             | 38       |
|    time_elapsed    | 4175     |
|    total_timesteps | 160780   |
| train/             |          |
|    actor_loss      | -89.2    |
|    critic_loss     | 1.98     |
|    ent_coef        | 0.0102   |
|    ent_coef_loss   | 0.0147   |
|    learning_rate   | 0.0003   |
|    n_updates       | 160679   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 440      |
|    fps             | 38       |
|    time_elapsed    | 4245     |
|    total_timesteps | 163426   |
| train/             |          |
|    actor_loss      | -88.9    |
|    critic_loss     | 1.71     |
|    ent_coef        | 0.01     |
|    ent_coef_loss   | -1.11    |
|    learning_rate   | 0.0003   |
|    n_updates       | 163325   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 444      |
|    fps             | 38       |
|    time_elapsed    | 4298     |
|    total_timesteps | 165527   |
| train/             |          |
|    actor_loss      | -89.6    |
|    critic_loss     | 3.11     |
|    ent_coef        | 0.00984  |
|    ent_coef_loss   | 0.956    |
|    learning_rate   | 0.0003   |
|    n_updates       | 165426   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 448      |
|    fps             | 38       |
|    time_elapsed    | 4351     |
|    total_timesteps | 167587   |
| train/             |          |
|    actor_loss      | -89      |
|    critic_loss     | 1.97     |
|    ent_coef        | 0.00994  |
|    ent_coef_loss   | -0.765   |
|    learning_rate   | 0.0003   |
|    n_updates       | 167486   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 452      |
|    fps             | 38       |
|    time_elapsed    | 4402     |
|    total_timesteps | 169570   |
| train/             |          |
|    actor_loss      | -88.9    |
|    critic_loss     | 1.47     |
|    ent_coef        | 0.01     |
|    ent_coef_loss   | -0.167   |
|    learning_rate   | 0.0003   |
|    n_updates       | 169469   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 456      |
|    fps             | 38       |
|    time_elapsed    | 4476     |
|    total_timesteps | 172620   |
| train/             |          |
|    actor_loss      | -90.9    |
|    critic_loss     | 1.96     |
|    ent_coef        | 0.00984  |
|    ent_coef_loss   | -0.614   |
|    learning_rate   | 0.0003   |
|    n_updates       | 172519   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 460      |
|    fps             | 38       |
|    time_elapsed    | 4533     |
|    total_timesteps | 174889   |
| train/             |          |
|    actor_loss      | -95.2    |
|    critic_loss     | 1.95     |
|    ent_coef        | 0.0098   |
|    ent_coef_loss   | 2.23     |
|    learning_rate   | 0.0003   |
|    n_updates       | 174788   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 464      |
|    fps             | 38       |
|    time_elapsed    | 4567     |
|    total_timesteps | 176327   |
| train/             |          |
|    actor_loss      | -95.3    |
|    critic_loss     | 2.76     |
|    ent_coef        | 0.00986  |
|    ent_coef_loss   | -0.166   |
|    learning_rate   | 0.0003   |
|    n_updates       | 176226   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 468      |
|    fps             | 38       |
|    time_elapsed    | 4625     |
|    total_timesteps | 178519   |
| train/             |          |
|    actor_loss      | -92.8    |
|    critic_loss     | 12       |
|    ent_coef        | 0.00975  |
|    ent_coef_loss   | 0.611    |
|    learning_rate   | 0.0003   |
|    n_updates       | 178418   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 472      |
|    fps             | 38       |
|    time_elapsed    | 4658     |
|    total_timesteps | 179716   |
| train/             |          |
|    actor_loss      | -91.9    |
|    critic_loss     | 4.69     |
|    ent_coef        | 0.01     |
|    ent_coef_loss   | -0.0185  |
|    learning_rate   | 0.0003   |
|    n_updates       | 179615   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 476      |
|    fps             | 38       |
|    time_elapsed    | 4674     |
|    total_timesteps | 180309   |
| train/             |          |
|    actor_loss      | -93.8    |
|    critic_loss     | 2.3      |
|    ent_coef        | 0.00977  |
|    ent_coef_loss   | 0.606    |
|    learning_rate   | 0.0003   |
|    n_updates       | 180208   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 480      |
|    fps             | 38       |
|    time_elapsed    | 4728     |
|    total_timesteps | 182345   |
| train/             |          |
|    actor_loss      | -90.7    |
|    critic_loss     | 4.97     |
|    ent_coef        | 0.00975  |
|    ent_coef_loss   | 0.324    |
|    learning_rate   | 0.0003   |
|    n_updates       | 182244   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 484      |
|    fps             | 38       |
|    time_elapsed    | 4792     |
|    total_timesteps | 184688   |
| train/             |          |
|    actor_loss      | -89      |
|    critic_loss     | 1.86     |
|    ent_coef        | 0.00979  |
|    ent_coef_loss   | 0.288    |
|    learning_rate   | 0.0003   |
|    n_updates       | 184587   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 488      |
|    fps             | 38       |
|    time_elapsed    | 4832     |
|    total_timesteps | 186100   |
| train/             |          |
|    actor_loss      | -91.7    |
|    critic_loss     | 2.01     |
|    ent_coef        | 0.00984  |
|    ent_coef_loss   | -0.101   |
|    learning_rate   | 0.0003   |
|    n_updates       | 185999   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 492      |
|    fps             | 38       |
|    time_elapsed    | 4914     |
|    total_timesteps | 189162   |
| train/             |          |
|    actor_loss      | -90.8    |
|    critic_loss     | 1.49     |
|    ent_coef        | 0.0101   |
|    ent_coef_loss   | -1.23    |
|    learning_rate   | 0.0003   |
|    n_updates       | 189061   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 496      |
|    fps             | 38       |
|    time_elapsed    | 4945     |
|    total_timesteps | 190307   |
| train/             |          |
|    actor_loss      | -92.4    |
|    critic_loss     | 2.07     |
|    ent_coef        | 0.00999  |
|    ent_coef_loss   | 0.224    |
|    learning_rate   | 0.0003   |
|    n_updates       | 190206   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 500      |
|    fps             | 38       |
|    time_elapsed    | 4989     |
|    total_timesteps | 191876   |
| train/             |          |
|    actor_loss      | -94      |
|    critic_loss     | 1.28     |
|    ent_coef        | 0.01     |
|    ent_coef_loss   | -1.61    |
|    learning_rate   | 0.0003   |
|    n_updates       | 191775   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 504      |
|    fps             | 38       |
|    time_elapsed    | 5044     |
|    total_timesteps | 193962   |
| train/             |          |
|    actor_loss      | -94.7    |
|    critic_loss     | 1.81     |
|    ent_coef        | 0.00984  |
|    ent_coef_loss   | 1.7      |
|    learning_rate   | 0.0003   |
|    n_updates       | 193861   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 508      |
|    fps             | 38       |
|    time_elapsed    | 5104     |
|    total_timesteps | 196175   |
| train/             |          |
|    actor_loss      | -95.3    |
|    critic_loss     | 1.84     |
|    ent_coef        | 0.00986  |
|    ent_coef_loss   | -0.858   |
|    learning_rate   | 0.0003   |
|    n_updates       | 196074   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 512      |
|    fps             | 38       |
|    time_elapsed    | 5158     |
|    total_timesteps | 198241   |
| train/             |          |
|    actor_loss      | -92.2    |
|    critic_loss     | 1.54     |
|    ent_coef        | 0.0098   |
|    ent_coef_loss   | -2.72    |
|    learning_rate   | 0.0003   |
|    n_updates       | 198140   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 516      |
|    fps             | 38       |
|    time_elapsed    | 5192     |
|    total_timesteps | 199477   |
| train/             |          |
|    actor_loss      | -93.2    |
|    critic_loss     | 2.83     |
|    ent_coef        | 0.01     |
|    ent_coef_loss   | -0.534   |
|    learning_rate   | 0.0003   |
|    n_updates       | 199376   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 520      |
|    fps             | 38       |
|    time_elapsed    | 5233     |
|    total_timesteps | 200953   |
| train/             |          |
|    actor_loss      | -95.1    |
|    critic_loss     | 1.44     |
|    ent_coef        | 0.00975  |
|    ent_coef_loss   | 0.233    |
|    learning_rate   | 0.0003   |
|    n_updates       | 200852   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 524      |
|    fps             | 38       |
|    time_elapsed    | 5243     |
|    total_timesteps | 201299   |
| train/             |          |
|    actor_loss      | -93.8    |
|    critic_loss     | 1.69     |
|    ent_coef        | 0.00992  |
|    ent_coef_loss   | -0.902   |
|    learning_rate   | 0.0003   |
|    n_updates       | 201198   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 528      |
|    fps             | 38       |
|    time_elapsed    | 5309     |
|    total_timesteps | 203698   |
| train/             |          |
|    actor_loss      | -97.9    |
|    critic_loss     | 2.11     |
|    ent_coef        | 0.00999  |
|    ent_coef_loss   | 1.3      |
|    learning_rate   | 0.0003   |
|    n_updates       | 203597   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 532      |
|    fps             | 38       |
|    time_elapsed    | 5342     |
|    total_timesteps | 204946   |
| train/             |          |
|    actor_loss      | -96      |
|    critic_loss     | 4.01     |
|    ent_coef        | 0.00979  |
|    ent_coef_loss   | 0.408    |
|    learning_rate   | 0.0003   |
|    n_updates       | 204845   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 536      |
|    fps             | 38       |
|    time_elapsed    | 5410     |
|    total_timesteps | 207336   |
| train/             |          |
|    actor_loss      | -92.7    |
|    critic_loss     | 1.7      |
|    ent_coef        | 0.01     |
|    ent_coef_loss   | -1.47    |
|    learning_rate   | 0.0003   |
|    n_updates       | 207235   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 540      |
|    fps             | 38       |
|    time_elapsed    | 5469     |
|    total_timesteps | 209592   |
| train/             |          |
|    actor_loss      | -99      |
|    critic_loss     | 4.27     |
|    ent_coef        | 0.0097   |
|    ent_coef_loss   | -0.976   |
|    learning_rate   | 0.0003   |
|    n_updates       | 209491   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 544      |
|    fps             | 38       |
|    time_elapsed    | 5549     |
|    total_timesteps | 212612   |
| train/             |          |
|    actor_loss      | -97.3    |
|    critic_loss     | 2.22     |
|    ent_coef        | 0.00998  |
|    ent_coef_loss   | 0.312    |
|    learning_rate   | 0.0003   |
|    n_updates       | 212511   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 548      |
|    fps             | 38       |
|    time_elapsed    | 5607     |
|    total_timesteps | 214692   |
| train/             |          |
|    actor_loss      | -96.5    |
|    critic_loss     | 1.75     |
|    ent_coef        | 0.01     |
|    ent_coef_loss   | -1.54    |
|    learning_rate   | 0.0003   |
|    n_updates       | 214591   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 552      |
|    fps             | 38       |
|    time_elapsed    | 5674     |
|    total_timesteps | 217139   |
| train/             |          |
|    actor_loss      | -101     |
|    critic_loss     | 2.38     |
|    ent_coef        | 0.0102   |
|    ent_coef_loss   | -0.444   |
|    learning_rate   | 0.0003   |
|    n_updates       | 217038   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 556      |
|    fps             | 38       |
|    time_elapsed    | 5731     |
|    total_timesteps | 219297   |
| train/             |          |
|    actor_loss      | -99.4    |
|    critic_loss     | 2.21     |
|    ent_coef        | 0.0101   |
|    ent_coef_loss   | 0.96     |
|    learning_rate   | 0.0003   |
|    n_updates       | 219196   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 560      |
|    fps             | 38       |
|    time_elapsed    | 5761     |
|    total_timesteps | 220468   |
| train/             |          |
|    actor_loss      | -103     |
|    critic_loss     | 1.82     |
|    ent_coef        | 0.0101   |
|    ent_coef_loss   | 0.632    |
|    learning_rate   | 0.0003   |
|    n_updates       | 220367   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 564      |
|    fps             | 38       |
|    time_elapsed    | 5822     |
|    total_timesteps | 222674   |
| train/             |          |
|    actor_loss      | -99.9    |
|    critic_loss     | 1.47     |
|    ent_coef        | 0.00993  |
|    ent_coef_loss   | 0.174    |
|    learning_rate   | 0.0003   |
|    n_updates       | 222573   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 568      |
|    fps             | 38       |
|    time_elapsed    | 5881     |
|    total_timesteps | 224891   |
| train/             |          |
|    actor_loss      | -100     |
|    critic_loss     | 1.85     |
|    ent_coef        | 0.00997  |
|    ent_coef_loss   | 1.26     |
|    learning_rate   | 0.0003   |
|    n_updates       | 224790   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 572      |
|    fps             | 38       |
|    time_elapsed    | 5962     |
|    total_timesteps | 227955   |
| train/             |          |
|    actor_loss      | -97.7    |
|    critic_loss     | 1.95     |
|    ent_coef        | 0.01     |
|    ent_coef_loss   | -2.02    |
|    learning_rate   | 0.0003   |
|    n_updates       | 227854   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 576      |
|    fps             | 38       |
|    time_elapsed    | 6020     |
|    total_timesteps | 230092   |
| train/             |          |
|    actor_loss      | -95.6    |
|    critic_loss     | 1.66     |
|    ent_coef        | 0.01     |
|    ent_coef_loss   | -0.745   |
|    learning_rate   | 0.0003   |
|    n_updates       | 229991   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 580      |
|    fps             | 38       |
|    time_elapsed    | 6080     |
|    total_timesteps | 232400   |
| train/             |          |
|    actor_loss      | -101     |
|    critic_loss     | 1.98     |
|    ent_coef        | 0.0102   |
|    ent_coef_loss   | 1.99     |
|    learning_rate   | 0.0003   |
|    n_updates       | 232299   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 584      |
|    fps             | 38       |
|    time_elapsed    | 6117     |
|    total_timesteps | 233758   |
| train/             |          |
|    actor_loss      | -99.9    |
|    critic_loss     | 6.24     |
|    ent_coef        | 0.0103   |
|    ent_coef_loss   | -1.1     |
|    learning_rate   | 0.0003   |
|    n_updates       | 233657   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 588      |
|    fps             | 38       |
|    time_elapsed    | 6200     |
|    total_timesteps | 236805   |
| train/             |          |
|    actor_loss      | -99.8    |
|    critic_loss     | 3.44     |
|    ent_coef        | 0.0106   |
|    ent_coef_loss   | 0.583    |
|    learning_rate   | 0.0003   |
|    n_updates       | 236704   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 592      |
|    fps             | 38       |
|    time_elapsed    | 6236     |
|    total_timesteps | 238123   |
| train/             |          |
|    actor_loss      | -103     |
|    critic_loss     | 1.89     |
|    ent_coef        | 0.0101   |
|    ent_coef_loss   | 0.99     |
|    learning_rate   | 0.0003   |
|    n_updates       | 238022   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 596      |
|    fps             | 38       |
|    time_elapsed    | 6299     |
|    total_timesteps | 240380   |
| train/             |          |
|    actor_loss      | -95.9    |
|    critic_loss     | 1.63     |
|    ent_coef        | 0.0104   |
|    ent_coef_loss   | -0.585   |
|    learning_rate   | 0.0003   |
|    n_updates       | 240279   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 600      |
|    fps             | 38       |
|    time_elapsed    | 6378     |
|    total_timesteps | 243164   |
| train/             |          |
|    actor_loss      | -101     |
|    critic_loss     | 2.14     |
|    ent_coef        | 0.0102   |
|    ent_coef_loss   | 0.199    |
|    learning_rate   | 0.0003   |
|    n_updates       | 243063   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 604      |
|    fps             | 38       |
|    time_elapsed    | 6463     |
|    total_timesteps | 246233   |
| train/             |          |
|    actor_loss      | -100     |
|    critic_loss     | 2.14     |
|    ent_coef        | 0.0104   |
|    ent_coef_loss   | 0.727    |
|    learning_rate   | 0.0003   |
|    n_updates       | 246132   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 608      |
|    fps             | 38       |
|    time_elapsed    | 6550     |
|    total_timesteps | 249497   |
| train/             |          |
|    actor_loss      | -103     |
|    critic_loss     | 2.64     |
|    ent_coef        | 0.0101   |
|    ent_coef_loss   | 0.12     |
|    learning_rate   | 0.0003   |
|    n_updates       | 249396   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 612      |
|    fps             | 38       |
|    time_elapsed    | 6613     |
|    total_timesteps | 251835   |
| train/             |          |
|    actor_loss      | -107     |
|    critic_loss     | 2.81     |
|    ent_coef        | 0.0104   |
|    ent_coef_loss   | 1.47     |
|    learning_rate   | 0.0003   |
|    n_updates       | 251734   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 616      |
|    fps             | 38       |
|    time_elapsed    | 6681     |
|    total_timesteps | 254240   |
| train/             |          |
|    actor_loss      | -103     |
|    critic_loss     | 2.34     |
|    ent_coef        | 0.0106   |
|    ent_coef_loss   | 0.44     |
|    learning_rate   | 0.0003   |
|    n_updates       | 254139   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 620      |
|    fps             | 38       |
|    time_elapsed    | 6748     |
|    total_timesteps | 256754   |
| train/             |          |
|    actor_loss      | -103     |
|    critic_loss     | 3.77     |
|    ent_coef        | 0.0105   |
|    ent_coef_loss   | 0.281    |
|    learning_rate   | 0.0003   |
|    n_updates       | 256653   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 624      |
|    fps             | 38       |
|    time_elapsed    | 6763     |
|    total_timesteps | 257248   |
| train/             |          |
|    actor_loss      | -99.7    |
|    critic_loss     | 2.91     |
|    ent_coef        | 0.0104   |
|    ent_coef_loss   | 0.578    |
|    learning_rate   | 0.0003   |
|    n_updates       | 257147   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 628      |
|    fps             | 38       |
|    time_elapsed    | 6825     |
|    total_timesteps | 259520   |
| train/             |          |
|    actor_loss      | -106     |
|    critic_loss     | 1.7      |
|    ent_coef        | 0.0105   |
|    ent_coef_loss   | 0.803    |
|    learning_rate   | 0.0003   |
|    n_updates       | 259419   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 632      |
|    fps             | 38       |
|    time_elapsed    | 6885     |
|    total_timesteps | 261784   |
| train/             |          |
|    actor_loss      | -104     |
|    critic_loss     | 3.08     |
|    ent_coef        | 0.0103   |
|    ent_coef_loss   | 1.43     |
|    learning_rate   | 0.0003   |
|    n_updates       | 261683   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 636      |
|    fps             | 38       |
|    time_elapsed    | 6925     |
|    total_timesteps | 263229   |
| train/             |          |
|    actor_loss      | -106     |
|    critic_loss     | 2.39     |
|    ent_coef        | 0.0106   |
|    ent_coef_loss   | -1.13    |
|    learning_rate   | 0.0003   |
|    n_updates       | 263128   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 640      |
|    fps             | 38       |
|    time_elapsed    | 6984     |
|    total_timesteps | 265441   |
| train/             |          |
|    actor_loss      | -107     |
|    critic_loss     | 4.2      |
|    ent_coef        | 0.0104   |
|    ent_coef_loss   | 0.275    |
|    learning_rate   | 0.0003   |
|    n_updates       | 265340   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 644      |
|    fps             | 38       |
|    time_elapsed    | 7026     |
|    total_timesteps | 267063   |
| train/             |          |
|    actor_loss      | -104     |
|    critic_loss     | 4.15     |
|    ent_coef        | 0.0105   |
|    ent_coef_loss   | 0.507    |
|    learning_rate   | 0.0003   |
|    n_updates       | 266962   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 648      |
|    fps             | 37       |
|    time_elapsed    | 7072     |
|    total_timesteps | 268729   |
| train/             |          |
|    actor_loss      | -106     |
|    critic_loss     | 2.77     |
|    ent_coef        | 0.0105   |
|    ent_coef_loss   | 1.13     |
|    learning_rate   | 0.0003   |
|    n_updates       | 268628   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 652      |
|    fps             | 37       |
|    time_elapsed    | 7107     |
|    total_timesteps | 270009   |
| train/             |          |
|    actor_loss      | -105     |
|    critic_loss     | 3.42     |
|    ent_coef        | 0.0103   |
|    ent_coef_loss   | 0.295    |
|    learning_rate   | 0.0003   |
|    n_updates       | 269908   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 656      |
|    fps             | 38       |
|    time_elapsed    | 7140     |
|    total_timesteps | 271644   |
| train/             |          |
|    actor_loss      | -103     |
|    critic_loss     | 3.6      |
|    ent_coef        | 0.0103   |
|    ent_coef_loss   | 0.224    |
|    learning_rate   | 0.0003   |
|    n_updates       | 271543   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 660      |
|    fps             | 38       |
|    time_elapsed    | 7186     |
|    total_timesteps | 273481   |
| train/             |          |
|    actor_loss      | -110     |
|    critic_loss     | 2.81     |
|    ent_coef        | 0.0104   |
|    ent_coef_loss   | -0.457   |
|    learning_rate   | 0.0003   |
|    n_updates       | 273380   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 664      |
|    fps             | 38       |
|    time_elapsed    | 7258     |
|    total_timesteps | 276356   |
| train/             |          |
|    actor_loss      | -108     |
|    critic_loss     | 3.72     |
|    ent_coef        | 0.0103   |
|    ent_coef_loss   | 1.08     |
|    learning_rate   | 0.0003   |
|    n_updates       | 276255   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 668      |
|    fps             | 38       |
|    time_elapsed    | 7314     |
|    total_timesteps | 278555   |
| train/             |          |
|    actor_loss      | -106     |
|    critic_loss     | 3.86     |
|    ent_coef        | 0.0105   |
|    ent_coef_loss   | 1.13     |
|    learning_rate   | 0.0003   |
|    n_updates       | 278454   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 672      |
|    fps             | 38       |
|    time_elapsed    | 7369     |
|    total_timesteps | 280876   |
| train/             |          |
|    actor_loss      | -105     |
|    critic_loss     | 5.31     |
|    ent_coef        | 0.011    |
|    ent_coef_loss   | 1.53     |
|    learning_rate   | 0.0003   |
|    n_updates       | 280775   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 676      |
|    fps             | 38       |
|    time_elapsed    | 7401     |
|    total_timesteps | 282120   |
| train/             |          |
|    actor_loss      | -101     |
|    critic_loss     | 1.94     |
|    ent_coef        | 0.0111   |
|    ent_coef_loss   | -2.05    |
|    learning_rate   | 0.0003   |
|    n_updates       | 282019   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 680      |
|    fps             | 38       |
|    time_elapsed    | 7454     |
|    total_timesteps | 284209   |
| train/             |          |
|    actor_loss      | -111     |
|    critic_loss     | 3.25     |
|    ent_coef        | 0.0114   |
|    ent_coef_loss   | 2.91     |
|    learning_rate   | 0.0003   |
|    n_updates       | 284108   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 684      |
|    fps             | 38       |
|    time_elapsed    | 7515     |
|    total_timesteps | 286664   |
| train/             |          |
|    actor_loss      | -113     |
|    critic_loss     | 2.46     |
|    ent_coef        | 0.0114   |
|    ent_coef_loss   | 0.246    |
|    learning_rate   | 0.0003   |
|    n_updates       | 286563   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 688      |
|    fps             | 38       |
|    time_elapsed    | 7588     |
|    total_timesteps | 289592   |
| train/             |          |
|    actor_loss      | -110     |
|    critic_loss     | 5.9      |
|    ent_coef        | 0.0117   |
|    ent_coef_loss   | 0.967    |
|    learning_rate   | 0.0003   |
|    n_updates       | 289491   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 692      |
|    fps             | 38       |
|    time_elapsed    | 7641     |
|    total_timesteps | 291617   |
| train/             |          |
|    actor_loss      | -112     |
|    critic_loss     | 2.93     |
|    ent_coef        | 0.0116   |
|    ent_coef_loss   | 0.095    |
|    learning_rate   | 0.0003   |
|    n_updates       | 291516   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 696      |
|    fps             | 38       |
|    time_elapsed    | 7680     |
|    total_timesteps | 293166   |
| train/             |          |
|    actor_loss      | -117     |
|    critic_loss     | 3.76     |
|    ent_coef        | 0.0116   |
|    ent_coef_loss   | -0.268   |
|    learning_rate   | 0.0003   |
|    n_updates       | 293065   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 700      |
|    fps             | 38       |
|    time_elapsed    | 7715     |
|    total_timesteps | 294665   |
| train/             |          |
|    actor_loss      | -119     |
|    critic_loss     | 3.67     |
|    ent_coef        | 0.0118   |
|    ent_coef_loss   | 1.88     |
|    learning_rate   | 0.0003   |
|    n_updates       | 294564   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 704      |
|    fps             | 38       |
|    time_elapsed    | 7761     |
|    total_timesteps | 296432   |
| train/             |          |
|    actor_loss      | -112     |
|    critic_loss     | 2.45     |
|    ent_coef        | 0.0119   |
|    ent_coef_loss   | -0.946   |
|    learning_rate   | 0.0003   |
|    n_updates       | 296331   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 708      |
|    fps             | 38       |
|    time_elapsed    | 7817     |
|    total_timesteps | 298630   |
| train/             |          |
|    actor_loss      | -122     |
|    critic_loss     | 4.54     |
|    ent_coef        | 0.0135   |
|    ent_coef_loss   | 2        |
|    learning_rate   | 0.0003   |
|    n_updates       | 298529   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 712      |
|    fps             | 38       |
|    time_elapsed    | 7880     |
|    total_timesteps | 301082   |
| train/             |          |
|    actor_loss      | -126     |
|    critic_loss     | 7.34     |
|    ent_coef        | 0.0133   |
|    ent_coef_loss   | -0.682   |
|    learning_rate   | 0.0003   |
|    n_updates       | 300981   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 716      |
|    fps             | 38       |
|    time_elapsed    | 7967     |
|    total_timesteps | 304644   |
| train/             |          |
|    actor_loss      | -131     |
|    critic_loss     | 4.95     |
|    ent_coef        | 0.0148   |
|    ent_coef_loss   | 1.34     |
|    learning_rate   | 0.0003   |
|    n_updates       | 304543   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 720      |
|    fps             | 38       |
|    time_elapsed    | 8054     |
|    total_timesteps | 308074   |
| train/             |          |
|    actor_loss      | -129     |
|    critic_loss     | 4.32     |
|    ent_coef        | 0.0131   |
|    ent_coef_loss   | -0.538   |
|    learning_rate   | 0.0003   |
|    n_updates       | 307973   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 724      |
|    fps             | 38       |
|    time_elapsed    | 8106     |
|    total_timesteps | 310266   |
| train/             |          |
|    actor_loss      | -129     |
|    critic_loss     | 3.22     |
|    ent_coef        | 0.0134   |
|    ent_coef_loss   | 0.431    |
|    learning_rate   | 0.0003   |
|    n_updates       | 310165   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 728      |
|    fps             | 38       |
|    time_elapsed    | 8145     |
|    total_timesteps | 311877   |
| train/             |          |
|    actor_loss      | -129     |
|    critic_loss     | 4.76     |
|    ent_coef        | 0.0135   |
|    ent_coef_loss   | -0.303   |
|    learning_rate   | 0.0003   |
|    n_updates       | 311776   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 732      |
|    fps             | 38       |
|    time_elapsed    | 8200     |
|    total_timesteps | 314174   |
| train/             |          |
|    actor_loss      | -130     |
|    critic_loss     | 4.52     |
|    ent_coef        | 0.0132   |
|    ent_coef_loss   | -0.79    |
|    learning_rate   | 0.0003   |
|    n_updates       | 314073   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 736      |
|    fps             | 38       |
|    time_elapsed    | 8260     |
|    total_timesteps | 316584   |
| train/             |          |
|    actor_loss      | -132     |
|    critic_loss     | 5.45     |
|    ent_coef        | 0.0131   |
|    ent_coef_loss   | 0.951    |
|    learning_rate   | 0.0003   |
|    n_updates       | 316483   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 740      |
|    fps             | 38       |
|    time_elapsed    | 8289     |
|    total_timesteps | 317814   |
| train/             |          |
|    actor_loss      | -127     |
|    critic_loss     | 4.61     |
|    ent_coef        | 0.0135   |
|    ent_coef_loss   | -3.23    |
|    learning_rate   | 0.0003   |
|    n_updates       | 317713   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 744      |
|    fps             | 38       |
|    time_elapsed    | 8373     |
|    total_timesteps | 321135   |
| train/             |          |
|    actor_loss      | -135     |
|    critic_loss     | 11       |
|    ent_coef        | 0.0137   |
|    ent_coef_loss   | 1.32     |
|    learning_rate   | 0.0003   |
|    n_updates       | 321034   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 748      |
|    fps             | 38       |
|    time_elapsed    | 8415     |
|    total_timesteps | 322566   |
| train/             |          |
|    actor_loss      | -134     |
|    critic_loss     | 5.62     |
|    ent_coef        | 0.0132   |
|    ent_coef_loss   | -0.16    |
|    learning_rate   | 0.0003   |
|    n_updates       | 322465   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 752      |
|    fps             | 38       |
|    time_elapsed    | 8449     |
|    total_timesteps | 323993   |
| train/             |          |
|    actor_loss      | -130     |
|    critic_loss     | 4.43     |
|    ent_coef        | 0.0133   |
|    ent_coef_loss   | -1.17    |
|    learning_rate   | 0.0003   |
|    n_updates       | 323892   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 756      |
|    fps             | 38       |
|    time_elapsed    | 8502     |
|    total_timesteps | 326238   |
| train/             |          |
|    actor_loss      | -139     |
|    critic_loss     | 42.7     |
|    ent_coef        | 0.0126   |
|    ent_coef_loss   | 2.13     |
|    learning_rate   | 0.0003   |
|    n_updates       | 326137   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 760      |
|    fps             | 38       |
|    time_elapsed    | 8554     |
|    total_timesteps | 328421   |
| train/             |          |
|    actor_loss      | -139     |
|    critic_loss     | 6.34     |
|    ent_coef        | 0.0129   |
|    ent_coef_loss   | 1.45     |
|    learning_rate   | 0.0003   |
|    n_updates       | 328320   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 764      |
|    fps             | 38       |
|    time_elapsed    | 8607     |
|    total_timesteps | 330554   |
| train/             |          |
|    actor_loss      | -136     |
|    critic_loss     | 5.06     |
|    ent_coef        | 0.0127   |
|    ent_coef_loss   | 0.697    |
|    learning_rate   | 0.0003   |
|    n_updates       | 330453   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 768      |
|    fps             | 38       |
|    time_elapsed    | 8709     |
|    total_timesteps | 334554   |
| train/             |          |
|    actor_loss      | -136     |
|    critic_loss     | 4.97     |
|    ent_coef        | 0.014    |
|    ent_coef_loss   | -0.0903  |
|    learning_rate   | 0.0003   |
|    n_updates       | 334453   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 772      |
|    fps             | 38       |
|    time_elapsed    | 8718     |
|    total_timesteps | 334941   |
| train/             |          |
|    actor_loss      | -131     |
|    critic_loss     | 4.28     |
|    ent_coef        | 0.0143   |
|    ent_coef_loss   | -1.9     |
|    learning_rate   | 0.0003   |
|    n_updates       | 334840   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 776      |
|    fps             | 38       |
|    time_elapsed    | 8750     |
|    total_timesteps | 336147   |
| train/             |          |
|    actor_loss      | -141     |
|    critic_loss     | 5.12     |
|    ent_coef        | 0.014    |
|    ent_coef_loss   | -2.67    |
|    learning_rate   | 0.0003   |
|    n_updates       | 336046   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 780      |
|    fps             | 38       |
|    time_elapsed    | 8808     |
|    total_timesteps | 338334   |
| train/             |          |
|    actor_loss      | -141     |
|    critic_loss     | 4.28     |
|    ent_coef        | 0.0135   |
|    ent_coef_loss   | -2.7     |
|    learning_rate   | 0.0003   |
|    n_updates       | 338233   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 784      |
|    fps             | 38       |
|    time_elapsed    | 8842     |
|    total_timesteps | 339561   |
| train/             |          |
|    actor_loss      | -144     |
|    critic_loss     | 9.16     |
|    ent_coef        | 0.0132   |
|    ent_coef_loss   | -0.771   |
|    learning_rate   | 0.0003   |
|    n_updates       | 339460   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 788      |
|    fps             | 38       |
|    time_elapsed    | 8921     |
|    total_timesteps | 342825   |
| train/             |          |
|    actor_loss      | -149     |
|    critic_loss     | 4.63     |
|    ent_coef        | 0.0132   |
|    ent_coef_loss   | 0.198    |
|    learning_rate   | 0.0003   |
|    n_updates       | 342724   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 792      |
|    fps             | 38       |
|    time_elapsed    | 9022     |
|    total_timesteps | 346825   |
| train/             |          |
|    actor_loss      | -144     |
|    critic_loss     | 5.23     |
|    ent_coef        | 0.013    |
|    ent_coef_loss   | 0.735    |
|    learning_rate   | 0.0003   |
|    n_updates       | 346724   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 796      |
|    fps             | 38       |
|    time_elapsed    | 9076     |
|    total_timesteps | 348966   |
| train/             |          |
|    actor_loss      | -145     |
|    critic_loss     | 3.87     |
|    ent_coef        | 0.0129   |
|    ent_coef_loss   | 0.0324   |
|    learning_rate   | 0.0003   |
|    n_updates       | 348865   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 800      |
|    fps             | 38       |
|    time_elapsed    | 9180     |
|    total_timesteps | 352966   |
| train/             |          |
|    actor_loss      | -143     |
|    critic_loss     | 4.57     |
|    ent_coef        | 0.013    |
|    ent_coef_loss   | -0.66    |
|    learning_rate   | 0.0003   |
|    n_updates       | 352865   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 804      |
|    fps             | 38       |
|    time_elapsed    | 9235     |
|    total_timesteps | 355064   |
| train/             |          |
|    actor_loss      | -139     |
|    critic_loss     | 11       |
|    ent_coef        | 0.0132   |
|    ent_coef_loss   | 0.512    |
|    learning_rate   | 0.0003   |
|    n_updates       | 354963   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 808      |
|    fps             | 38       |
|    time_elapsed    | 9290     |
|    total_timesteps | 357225   |
| train/             |          |
|    actor_loss      | -146     |
|    critic_loss     | 5.5      |
|    ent_coef        | 0.0131   |
|    ent_coef_loss   | 0.36     |
|    learning_rate   | 0.0003   |
|    n_updates       | 357124   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 812      |
|    fps             | 38       |
|    time_elapsed    | 9366     |
|    total_timesteps | 360261   |
| train/             |          |
|    actor_loss      | -141     |
|    critic_loss     | 4.7      |
|    ent_coef        | 0.0128   |
|    ent_coef_loss   | -0.95    |
|    learning_rate   | 0.0003   |
|    n_updates       | 360160   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 816      |
|    fps             | 38       |
|    time_elapsed    | 9419     |
|    total_timesteps | 362318   |
| train/             |          |
|    actor_loss      | -145     |
|    critic_loss     | 5.84     |
|    ent_coef        | 0.0129   |
|    ent_coef_loss   | 1.78     |
|    learning_rate   | 0.0003   |
|    n_updates       | 362217   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 820      |
|    fps             | 38       |
|    time_elapsed    | 9474     |
|    total_timesteps | 364463   |
| train/             |          |
|    actor_loss      | -138     |
|    critic_loss     | 6.1      |
|    ent_coef        | 0.0126   |
|    ent_coef_loss   | -0.208   |
|    learning_rate   | 0.0003   |
|    n_updates       | 364362   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 824      |
|    fps             | 38       |
|    time_elapsed    | 9531     |
|    total_timesteps | 366792   |
| train/             |          |
|    actor_loss      | -137     |
|    critic_loss     | 4.15     |
|    ent_coef        | 0.0131   |
|    ent_coef_loss   | -0.211   |
|    learning_rate   | 0.0003   |
|    n_updates       | 366691   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 828      |
|    fps             | 38       |
|    time_elapsed    | 9628     |
|    total_timesteps | 370537   |
| train/             |          |
|    actor_loss      | -131     |
|    critic_loss     | 4.58     |
|    ent_coef        | 0.0132   |
|    ent_coef_loss   | -1.07    |
|    learning_rate   | 0.0003   |
|    n_updates       | 370436   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 832      |
|    fps             | 38       |
|    time_elapsed    | 9729     |
|    total_timesteps | 374374   |
| train/             |          |
|    actor_loss      | -137     |
|    critic_loss     | 4.05     |
|    ent_coef        | 0.0127   |
|    ent_coef_loss   | -0.612   |
|    learning_rate   | 0.0003   |
|    n_updates       | 374273   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 836      |
|    fps             | 38       |
|    time_elapsed    | 9781     |
|    total_timesteps | 376486   |
| train/             |          |
|    actor_loss      | -139     |
|    critic_loss     | 5.1      |
|    ent_coef        | 0.0125   |
|    ent_coef_loss   | -0.525   |
|    learning_rate   | 0.0003   |
|    n_updates       | 376385   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 840      |
|    fps             | 38       |
|    time_elapsed    | 9832     |
|    total_timesteps | 378616   |
| train/             |          |
|    actor_loss      | -132     |
|    critic_loss     | 2.98     |
|    ent_coef        | 0.0128   |
|    ent_coef_loss   | 0.203    |
|    learning_rate   | 0.0003   |
|    n_updates       | 378515   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 844      |
|    fps             | 38       |
|    time_elapsed    | 9877     |
|    total_timesteps | 380444   |
| train/             |          |
|    actor_loss      | -129     |
|    critic_loss     | 5.64     |
|    ent_coef        | 0.0132   |
|    ent_coef_loss   | -0.47    |
|    learning_rate   | 0.0003   |
|    n_updates       | 380343   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 848      |
|    fps             | 38       |
|    time_elapsed    | 9962     |
|    total_timesteps | 383421   |
| train/             |          |
|    actor_loss      | -127     |
|    critic_loss     | 2.21     |
|    ent_coef        | 0.0126   |
|    ent_coef_loss   | 0.0829   |
|    learning_rate   | 0.0003   |
|    n_updates       | 383320   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 852      |
|    fps             | 38       |
|    time_elapsed    | 10020    |
|    total_timesteps | 385530   |
| train/             |          |
|    actor_loss      | -126     |
|    critic_loss     | 2.02     |
|    ent_coef        | 0.0125   |
|    ent_coef_loss   | -0.832   |
|    learning_rate   | 0.0003   |
|    n_updates       | 385429   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 856      |
|    fps             | 38       |
|    time_elapsed    | 10092    |
|    total_timesteps | 388162   |
| train/             |          |
|    actor_loss      | -129     |
|    critic_loss     | 20.5     |
|    ent_coef        | 0.0128   |
|    ent_coef_loss   | 0.818    |
|    learning_rate   | 0.0003   |
|    n_updates       | 388061   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 860      |
|    fps             | 38       |
|    time_elapsed    | 10136    |
|    total_timesteps | 389801   |
| train/             |          |
|    actor_loss      | -129     |
|    critic_loss     | 6.15     |
|    ent_coef        | 0.0128   |
|    ent_coef_loss   | -0.775   |
|    learning_rate   | 0.0003   |
|    n_updates       | 389700   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 864      |
|    fps             | 38       |
|    time_elapsed    | 10169    |
|    total_timesteps | 391004   |
| train/             |          |
|    actor_loss      | -128     |
|    critic_loss     | 2.89     |
|    ent_coef        | 0.0129   |
|    ent_coef_loss   | 0.668    |
|    learning_rate   | 0.0003   |
|    n_updates       | 390903   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 868      |
|    fps             | 38       |
|    time_elapsed    | 10206    |
|    total_timesteps | 392391   |
| train/             |          |
|    actor_loss      | -122     |
|    critic_loss     | 4.08     |
|    ent_coef        | 0.0126   |
|    ent_coef_loss   | -0.529   |
|    learning_rate   | 0.0003   |
|    n_updates       | 392290   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 872      |
|    fps             | 38       |
|    time_elapsed    | 10292    |
|    total_timesteps | 395522   |
| train/             |          |
|    actor_loss      | -131     |
|    critic_loss     | 4.69     |
|    ent_coef        | 0.0125   |
|    ent_coef_loss   | -0.173   |
|    learning_rate   | 0.0003   |
|    n_updates       | 395421   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 876      |
|    fps             | 38       |
|    time_elapsed    | 10374    |
|    total_timesteps | 398547   |
| train/             |          |
|    actor_loss      | -123     |
|    critic_loss     | 4.67     |
|    ent_coef        | 0.0124   |
|    ent_coef_loss   | -1.29    |
|    learning_rate   | 0.0003   |
|    n_updates       | 398446   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 880      |
|    fps             | 38       |
|    time_elapsed    | 10403    |
|    total_timesteps | 399614   |
| train/             |          |
|    actor_loss      | -124     |
|    critic_loss     | 3.37     |
|    ent_coef        | 0.0123   |
|    ent_coef_loss   | -0.0217  |
|    learning_rate   | 0.0003   |
|    n_updates       | 399513   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 884      |
|    fps             | 38       |
|    time_elapsed    | 10492    |
|    total_timesteps | 402862   |
| train/             |          |
|    actor_loss      | -127     |
|    critic_loss     | 2        |
|    ent_coef        | 0.0125   |
|    ent_coef_loss   | 1.4      |
|    learning_rate   | 0.0003   |
|    n_updates       | 402761   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 888      |
|    fps             | 38       |
|    time_elapsed    | 10512    |
|    total_timesteps | 403545   |
| train/             |          |
|    actor_loss      | -128     |
|    critic_loss     | 3.6      |
|    ent_coef        | 0.0123   |
|    ent_coef_loss   | -1.1     |
|    learning_rate   | 0.0003   |
|    n_updates       | 403444   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 892      |
|    fps             | 38       |
|    time_elapsed    | 10569    |
|    total_timesteps | 405692   |
| train/             |          |
|    actor_loss      | -127     |
|    critic_loss     | 3.37     |
|    ent_coef        | 0.013    |
|    ent_coef_loss   | -1.28    |
|    learning_rate   | 0.0003   |
|    n_updates       | 405591   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 896      |
|    fps             | 38       |
|    time_elapsed    | 10601    |
|    total_timesteps | 406920   |
| train/             |          |
|    actor_loss      | -123     |
|    critic_loss     | 4.81     |
|    ent_coef        | 0.0126   |
|    ent_coef_loss   | 0.085    |
|    learning_rate   | 0.0003   |
|    n_updates       | 406819   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 900      |
|    fps             | 38       |
|    time_elapsed    | 10659    |
|    total_timesteps | 409118   |
| train/             |          |
|    actor_loss      | -127     |
|    critic_loss     | 3.86     |
|    ent_coef        | 0.0123   |
|    ent_coef_loss   | -0.125   |
|    learning_rate   | 0.0003   |
|    n_updates       | 409017   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 904      |
|    fps             | 38       |
|    time_elapsed    | 10745    |
|    total_timesteps | 412376   |
| train/             |          |
|    actor_loss      | -123     |
|    critic_loss     | 2.49     |
|    ent_coef        | 0.0125   |
|    ent_coef_loss   | -0.16    |
|    learning_rate   | 0.0003   |
|    n_updates       | 412275   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 908      |
|    fps             | 38       |
|    time_elapsed    | 10813    |
|    total_timesteps | 414978   |
| train/             |          |
|    actor_loss      | -127     |
|    critic_loss     | 5.62     |
|    ent_coef        | 0.0127   |
|    ent_coef_loss   | 0.369    |
|    learning_rate   | 0.0003   |
|    n_updates       | 414877   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 912      |
|    fps             | 38       |
|    time_elapsed    | 10849    |
|    total_timesteps | 416298   |
| train/             |          |
|    actor_loss      | -120     |
|    critic_loss     | 3.72     |
|    ent_coef        | 0.0128   |
|    ent_coef_loss   | -0.517   |
|    learning_rate   | 0.0003   |
|    n_updates       | 416197   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 916      |
|    fps             | 38       |
|    time_elapsed    | 10889    |
|    total_timesteps | 417786   |
| train/             |          |
|    actor_loss      | -122     |
|    critic_loss     | 3.7      |
|    ent_coef        | 0.0129   |
|    ent_coef_loss   | -0.0717  |
|    learning_rate   | 0.0003   |
|    n_updates       | 417685   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 920      |
|    fps             | 38       |
|    time_elapsed    | 10907    |
|    total_timesteps | 418353   |
| train/             |          |
|    actor_loss      | -123     |
|    critic_loss     | 3.35     |
|    ent_coef        | 0.0125   |
|    ent_coef_loss   | 0.0533   |
|    learning_rate   | 0.0003   |
|    n_updates       | 418252   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 924      |
|    fps             | 38       |
|    time_elapsed    | 10914    |
|    total_timesteps | 418600   |
| train/             |          |
|    actor_loss      | -122     |
|    critic_loss     | 3.79     |
|    ent_coef        | 0.0126   |
|    ent_coef_loss   | -0.978   |
|    learning_rate   | 0.0003   |
|    n_updates       | 418499   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 928      |
|    fps             | 38       |
|    time_elapsed    | 10988    |
|    total_timesteps | 421382   |
| train/             |          |
|    actor_loss      | -123     |
|    critic_loss     | 2.68     |
|    ent_coef        | 0.0125   |
|    ent_coef_loss   | -1.2     |
|    learning_rate   | 0.0003   |
|    n_updates       | 421281   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 932      |
|    fps             | 38       |
|    time_elapsed    | 11040    |
|    total_timesteps | 423295   |
| train/             |          |
|    actor_loss      | -122     |
|    critic_loss     | 4.02     |
|    ent_coef        | 0.0126   |
|    ent_coef_loss   | -0.708   |
|    learning_rate   | 0.0003   |
|    n_updates       | 423194   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 936      |
|    fps             | 38       |
|    time_elapsed    | 11078    |
|    total_timesteps | 424733   |
| train/             |          |
|    actor_loss      | -123     |
|    critic_loss     | 3.93     |
|    ent_coef        | 0.0123   |
|    ent_coef_loss   | -1.5     |
|    learning_rate   | 0.0003   |
|    n_updates       | 424632   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 940      |
|    fps             | 38       |
|    time_elapsed    | 11164    |
|    total_timesteps | 427980   |
| train/             |          |
|    actor_loss      | -121     |
|    critic_loss     | 6.6      |
|    ent_coef        | 0.0127   |
|    ent_coef_loss   | -0.365   |
|    learning_rate   | 0.0003   |
|    n_updates       | 427879   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 944      |
|    fps             | 38       |
|    time_elapsed    | 11223    |
|    total_timesteps | 430198   |
| train/             |          |
|    actor_loss      | -131     |
|    critic_loss     | 3.85     |
|    ent_coef        | 0.0123   |
|    ent_coef_loss   | 0.967    |
|    learning_rate   | 0.0003   |
|    n_updates       | 430097   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 948      |
|    fps             | 38       |
|    time_elapsed    | 11260    |
|    total_timesteps | 431565   |
| train/             |          |
|    actor_loss      | -122     |
|    critic_loss     | 4.84     |
|    ent_coef        | 0.0125   |
|    ent_coef_loss   | -0.191   |
|    learning_rate   | 0.0003   |
|    n_updates       | 431464   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 952      |
|    fps             | 38       |
|    time_elapsed    | 11341    |
|    total_timesteps | 434614   |
| train/             |          |
|    actor_loss      | -112     |
|    critic_loss     | 10.7     |
|    ent_coef        | 0.0126   |
|    ent_coef_loss   | 0.169    |
|    learning_rate   | 0.0003   |
|    n_updates       | 434513   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 956      |
|    fps             | 38       |
|    time_elapsed    | 11406    |
|    total_timesteps | 436946   |
| train/             |          |
|    actor_loss      | -128     |
|    critic_loss     | 8.91     |
|    ent_coef        | 0.0126   |
|    ent_coef_loss   | 0.743    |
|    learning_rate   | 0.0003   |
|    n_updates       | 436845   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 960      |
|    fps             | 38       |
|    time_elapsed    | 11486    |
|    total_timesteps | 439983   |
| train/             |          |
|    actor_loss      | -117     |
|    critic_loss     | 2.48     |
|    ent_coef        | 0.0129   |
|    ent_coef_loss   | -1.57    |
|    learning_rate   | 0.0003   |
|    n_updates       | 439882   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 964      |
|    fps             | 38       |
|    time_elapsed    | 11523    |
|    total_timesteps | 441335   |
| train/             |          |
|    actor_loss      | -124     |
|    critic_loss     | 3.53     |
|    ent_coef        | 0.0127   |
|    ent_coef_loss   | -0.233   |
|    learning_rate   | 0.0003   |
|    n_updates       | 441234   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 968      |
|    fps             | 38       |
|    time_elapsed    | 11555    |
|    total_timesteps | 442529   |
| train/             |          |
|    actor_loss      | -120     |
|    critic_loss     | 2.97     |
|    ent_coef        | 0.0128   |
|    ent_coef_loss   | -1.48    |
|    learning_rate   | 0.0003   |
|    n_updates       | 442428   |
---------------------------------
----------------------------------
| time/              |           |
|    episodes        | 972       |
|    fps             | 38        |
|    time_elapsed    | 11616     |
|    total_timesteps | 444861    |
| train/             |           |
|    actor_loss      | -123      |
|    critic_loss     | 4.44      |
|    ent_coef        | 0.0124    |
|    ent_coef_loss   | -0.000727 |
|    learning_rate   | 0.0003    |
|    n_updates       | 444760    |
----------------------------------
---------------------------------
| time/              |          |
|    episodes        | 976      |
|    fps             | 38       |
|    time_elapsed    | 11657    |
|    total_timesteps | 446391   |
| train/             |          |
|    actor_loss      | -125     |
|    critic_loss     | 3.76     |
|    ent_coef        | 0.0128   |
|    ent_coef_loss   | 2.22     |
|    learning_rate   | 0.0003   |
|    n_updates       | 446290   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 980      |
|    fps             | 38       |
|    time_elapsed    | 11670    |
|    total_timesteps | 446849   |
| train/             |          |
|    actor_loss      | -123     |
|    critic_loss     | 2.65     |
|    ent_coef        | 0.0129   |
|    ent_coef_loss   | -0.685   |
|    learning_rate   | 0.0003   |
|    n_updates       | 446748   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 984      |
|    fps             | 38       |
|    time_elapsed    | 11705    |
|    total_timesteps | 448228   |
| train/             |          |
|    actor_loss      | -122     |
|    critic_loss     | 7.67     |
|    ent_coef        | 0.0128   |
|    ent_coef_loss   | 0.574    |
|    learning_rate   | 0.0003   |
|    n_updates       | 448127   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 988      |
|    fps             | 38       |
|    time_elapsed    | 11763    |
|    total_timesteps | 450419   |
| train/             |          |
|    actor_loss      | -125     |
|    critic_loss     | 3.8      |
|    ent_coef        | 0.0128   |
|    ent_coef_loss   | -0.665   |
|    learning_rate   | 0.0003   |
|    n_updates       | 450318   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 992      |
|    fps             | 38       |
|    time_elapsed    | 11793    |
|    total_timesteps | 451649   |
| train/             |          |
|    actor_loss      | -133     |
|    critic_loss     | 6.85     |
|    ent_coef        | 0.013    |
|    ent_coef_loss   | 0.516    |
|    learning_rate   | 0.0003   |
|    n_updates       | 451548   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 996      |
|    fps             | 38       |
|    time_elapsed    | 11827    |
|    total_timesteps | 453041   |
| train/             |          |
|    actor_loss      | -124     |
|    critic_loss     | 5.77     |
|    ent_coef        | 0.0131   |
|    ent_coef_loss   | 2.1      |
|    learning_rate   | 0.0003   |
|    n_updates       | 452940   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 1000     |
|    fps             | 38       |
|    time_elapsed    | 11907    |
|    total_timesteps | 456330   |
| train/             |          |
|    actor_loss      | -122     |
|    critic_loss     | 6.4      |
|    ent_coef        | 0.0127   |
|    ent_coef_loss   | -0.452   |
|    learning_rate   | 0.0003   |
|    n_updates       | 456229   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 1004     |
|    fps             | 38       |
|    time_elapsed    | 11972    |
|    total_timesteps | 459102   |
| train/             |          |
|    actor_loss      | -130     |
|    critic_loss     | 5.21     |
|    ent_coef        | 0.0126   |
|    ent_coef_loss   | 0.642    |
|    learning_rate   | 0.0003   |
|    n_updates       | 459001   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 1008     |
|    fps             | 38       |
|    time_elapsed    | 12011    |
|    total_timesteps | 460709   |
| train/             |          |
|    actor_loss      | -122     |
|    critic_loss     | 2.64     |
|    ent_coef        | 0.0133   |
|    ent_coef_loss   | -0.884   |
|    learning_rate   | 0.0003   |
|    n_updates       | 460608   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 1012     |
|    fps             | 38       |
|    time_elapsed    | 12087    |
|    total_timesteps | 463804   |
| train/             |          |
|    actor_loss      | -129     |
|    critic_loss     | 3.97     |
|    ent_coef        | 0.013    |
|    ent_coef_loss   | 0.515    |
|    learning_rate   | 0.0003   |
|    n_updates       | 463703   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 1016     |
|    fps             | 38       |
|    time_elapsed    | 12150    |
|    total_timesteps | 466468   |
| train/             |          |
|    actor_loss      | -128     |
|    critic_loss     | 4.89     |
|    ent_coef        | 0.0134   |
|    ent_coef_loss   | -1.04    |
|    learning_rate   | 0.0003   |
|    n_updates       | 466367   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 1020     |
|    fps             | 38       |
|    time_elapsed    | 12179    |
|    total_timesteps | 467733   |
| train/             |          |
|    actor_loss      | -126     |
|    critic_loss     | 2.73     |
|    ent_coef        | 0.0131   |
|    ent_coef_loss   | -0.219   |
|    learning_rate   | 0.0003   |
|    n_updates       | 467632   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 1024     |
|    fps             | 38       |
|    time_elapsed    | 12247    |
|    total_timesteps | 470525   |
| train/             |          |
|    actor_loss      | -127     |
|    critic_loss     | 4.11     |
|    ent_coef        | 0.0128   |
|    ent_coef_loss   | -0.495   |
|    learning_rate   | 0.0003   |
|    n_updates       | 470424   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 1028     |
|    fps             | 38       |
|    time_elapsed    | 12331    |
|    total_timesteps | 474143   |
| train/             |          |
|    actor_loss      | -125     |
|    critic_loss     | 4.62     |
|    ent_coef        | 0.0132   |
|    ent_coef_loss   | -0.0928  |
|    learning_rate   | 0.0003   |
|    n_updates       | 474042   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 1032     |
|    fps             | 38       |
|    time_elapsed    | 12403    |
|    total_timesteps | 477590   |
| train/             |          |
|    actor_loss      | -119     |
|    critic_loss     | 10.5     |
|    ent_coef        | 0.0134   |
|    ent_coef_loss   | -1.41    |
|    learning_rate   | 0.0003   |
|    n_updates       | 477489   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 1036     |
|    fps             | 38       |
|    time_elapsed    | 12441    |
|    total_timesteps | 479340   |
| train/             |          |
|    actor_loss      | -134     |
|    critic_loss     | 3.16     |
|    ent_coef        | 0.0132   |
|    ent_coef_loss   | 1.55     |
|    learning_rate   | 0.0003   |
|    n_updates       | 479239   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 1040     |
|    fps             | 38       |
|    time_elapsed    | 12500    |
|    total_timesteps | 482104   |
| train/             |          |
|    actor_loss      | -118     |
|    critic_loss     | 9.57     |
|    ent_coef        | 0.0131   |
|    ent_coef_loss   | -1.55    |
|    learning_rate   | 0.0003   |
|    n_updates       | 482003   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 1044     |
|    fps             | 38       |
|    time_elapsed    | 12582    |
|    total_timesteps | 485517   |
| train/             |          |
|    actor_loss      | -132     |
|    critic_loss     | 4.36     |
|    ent_coef        | 0.0131   |
|    ent_coef_loss   | 2.21     |
|    learning_rate   | 0.0003   |
|    n_updates       | 485416   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 1048     |
|    fps             | 38       |
|    time_elapsed    | 12642    |
|    total_timesteps | 488402   |
| train/             |          |
|    actor_loss      | -126     |
|    critic_loss     | 7.71     |
|    ent_coef        | 0.0132   |
|    ent_coef_loss   | 0.607    |
|    learning_rate   | 0.0003   |
|    n_updates       | 488301   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 1052     |
|    fps             | 38       |
|    time_elapsed    | 12690    |
|    total_timesteps | 490674   |
| train/             |          |
|    actor_loss      | -136     |
|    critic_loss     | 7.61     |
|    ent_coef        | 0.0132   |
|    ent_coef_loss   | 2.26     |
|    learning_rate   | 0.0003   |
|    n_updates       | 490573   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 1056     |
|    fps             | 38       |
|    time_elapsed    | 12743    |
|    total_timesteps | 493189   |
| train/             |          |
|    actor_loss      | -134     |
|    critic_loss     | 3.04     |
|    ent_coef        | 0.0132   |
|    ent_coef_loss   | -0.403   |
|    learning_rate   | 0.0003   |
|    n_updates       | 493088   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 1060     |
|    fps             | 38       |
|    time_elapsed    | 12793    |
|    total_timesteps | 496490   |
| train/             |          |
|    actor_loss      | -122     |
|    critic_loss     | 2.64     |
|    ent_coef        | 0.0135   |
|    ent_coef_loss   | -1.71    |
|    learning_rate   | 0.0003   |
|    n_updates       | 496389   |
---------------------------------
---------------------------------
| time/              |          |
|    episodes        | 1064     |
|    fps             | 38       |
|    time_elapsed    | 12834    |
|    total_timesteps | 499152   |
| train/             |          |
|    actor_loss      | -134     |
|    critic_loss     | 2.98     |
|    ent_coef        | 0.0132   |
|    ent_coef_loss   | 0.0845   |
|    learning_rate   | 0.0003   |
|    n_updates       | 499051   |
---------------------------------
/home/zhushaoq/.conda/envs/SB_FVRL/lib/python3.7/site-packages/glfw/__init__.py:916: GLFWError: (65544) b'X11: The DISPLAY environment variable is missing'
  warnings.warn(message, GLFWError)
/home/zhushaoq/.conda/envs/SB_FVRL/lib/python3.7/site-packages/torch/utils/tensorboard/__init__.py:5: DeprecationWarning: distutils Version classes are deprecated. Use packaging.version instead.
  tensorboard.__version__
/home/zhushaoq/.conda/envs/SB_FVRL/lib/python3.7/site-packages/torch/utils/tensorboard/__init__.py:6: DeprecationWarning: distutils Version classes are deprecated. Use packaging.version instead.
  ) < LooseVersion("1.15"):
