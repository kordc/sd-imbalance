# Results

## Trainings and evaluations
| Method Name                          | Test Accuracy | Test Balanced Accuracy | Airplane | Automobile | Bird  | Cat   | Deer  | Dog   | Frog  | Horse | Ship  | Truck |
|--------------------------------------|---------------|-------------------------|----------|------------|-------|-------|-------|-------|-------|-------|-------|-------|
| Full data + more augmentations      | 0.105         | 0.105                   | 0.904    | 0.000      | 0.022 | 0.006 | 0.004 | 0.016 | 0.003 | 0.065 | 0.022 | 0.010 |
| Downsampled + random-oversample + more augmentations | 0.147 | 0.148         | 0.333    | 0.000      | 0.245 | 0.000 | 0.008 | 0.252 | 0.002 | 0.004 | 0.572 | 0.051 |
| Downsampled + more augmentations                 | 0.150         | 0.148                   | 0.759    | 0.140      | 0.157 | 0.000 | 0.006 | 0.031 | 0.016 | 0.018 | 0.264 | 0.111 |
| Downsampled + random-undersample    | 0.217         | 0.217                   | 0.390    | 0.118      | 0.003 | 0.119 | 0.344 | 0.362 | 0.122 | 0.038 | 0.320 | 0.357 |
| Downsampled + class weighting       | 0.766         | 0.764                   | 0.859    | 0.892      | 0.744 | 0.041 | 0.825 | 0.774 | 0.885 | 0.866 | 0.908 | 0.864 |
| Downsampled                         | 0.807         | 0.807                   | 0.890    | 0.926      | 0.835 | 0.000 | 0.874 | 0.879 | 0.928 | 0.899 | 0.928 | 0.915 |
| Downsampled + label smoothing       | 0.808         | 0.807                   | 0.896    | 0.918      | 0.854 | 0.000 | 0.872 | 0.867 | 0.922 | 0.908 | 0.928 | 0.917 |
| Downsampled + SMOTE      | 0.811         | 0.809                   | 0.900    | 0.929      | 0.845 | 0.012 | 0.900 | 0.865 | 0.930 | 0.900 | 0.925 | 0.899 |
| Downsampled + random-oversample    | 0.812         | 0.812                   | 0.903    | 0.938      | 0.855 | 0.008 | 0.879 | 0.866 | 0.923 | 0.908 | 0.932 | 0.913 |
| Downsampled + ADASYN     | 0.813         | 0.812                   | 0.920    | 0.922      | 0.851 | 0.012 | 0.885 | 0.865 | 0.921 | 0.914 | 0.922 | 0.917 |
| Full data                           | 0.855         | 0.853                   | 0.870    | 0.921      | 0.820 | 0.707 | 0.865 | 0.778 | 0.897 | 0.868 | 0.924 | 0.904 |

## Data generation
1. Prompt
```py
f"{angle} of a {breed} cat {preposition} the {furniture}, {gaze}. The cat has realistic fur textures, intricate details, and sharp features, with soft lighting and a clear focus. The image has a shallow depth of field, emphasizing the cat in fine detail. 8k, cinematic, photorealistic"
```
- Cat breeds: 100 (np British Longhair)
- Prepositions: 6 (np on)
- Objects: 61 (np sofa)
- camera_angles: 5 (np a photo taken from above”)
- gaze_directions: 12 (np looking up)
- Użyty BeautifulPrompt (BeautifulPrompt: Towards Automatic Prompt Engineering for Text-to-Image Synthesis)
- Przykład: a photo taken from behind of a American Wirehair cat next to the cabinet, looking to the right. The cat has realistic fur textures, intricate details, and sharp features, with soft lighting and a clear focus. The image has a shallow depth of field, emphasizing the cat in fine detail. 8k, cinematic, photorealistic

2. Wybrane modele do ewaluacji:
- stable-diffusion-3.5-large-turbo - 1.5h na 6k zdjęć
- black-forest-labs/FLUX.1-schnell - 2.5h na 6k zdjęć
- black-forest-labs/FLUX.1-dev - 14.5h na 6k zdjęć


Wszystko liczone jest na maszynie Lambda z NVIDIA A100, 40GB RAM, 8vCPU, 1TB SSD, 1Gbps