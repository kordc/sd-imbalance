# Results

## Trainings and evaluations

## Trainings and evaluations

| Method Name                                              | Test Accuracy | Test Balanced Accuracy | Cat   |
|----------------------------------------------------------|---------------|------------------------|-------|
| Full data + more augmentations                           | 0.105         | 0.105                  | 0.006 |
| Downsampled + random-oversample + more augmentations     | 0.147         | 0.148                  | 0.000 |
| Downsampled + more augmentations                         | 0.150         | 0.148                  | 0.000 |
| Downsampled + random-undersample                         | 0.217         | 0.217                  | 0.119 |
| Downsampled + class weighting                            | 0.766         | 0.764                  | 0.041 |
| Downsampled                                              | 0.807         | 0.807                  | 0.000 |
| Downsampled + label smoothing                            | 0.808         | 0.807                  | 0.000 |
| flux1.schnell                                            | 0.810         | 0.808                  | 0.022 |
| Downsampled + SMOTE                                      | 0.811         | 0.809                  | 0.012 |
| Downsampled + random-oversample                          | 0.812         | 0.812                  | 0.008 |
| Downsampled + ADASYN                                     | 0.813         | 0.812                  | 0.012 |
| FLUX1.dev                                                | 0.815         | 0.813                  | 0.039 |
| stable-diffusion-3.5-large-turbo                         | 0.817         | 0.817                  | 0.094 |
| Full data                                                | 0.855         | 0.853                  | 0.707 |



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