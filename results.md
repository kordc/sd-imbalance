# Results

## Trainings and evaluations

## Trainings and evaluations

| Method Name                                              | Test Accuracy | Test Balanced Accuracy | Cat   |
|----------------------------------------------------------|---------------|------------------------|-------|
| All classes downsampled to 1% + Full Synthetic           | 0.493         | N/A                    | 0.429 |
| Cat downsampled to 1%                                    | 0.764         | N/A                    | 0.036 |
| Cat downsampled to 1% + ADASYN                           | 0.803         | N/A                    | 0.010 |
| Cat downsampled to 1% + FLUX.1-Redux augmentation        | 0.815         | N/A                    | 0.073 |
| Cat downsampled to 1% + SD3.5L-Turbo                     | 0.817         | N/A                    | 0.094 |
| Cat downsampled to 1% + SDXL                             | 0.811         | N/A                    | 0.061 |
| Cat downsampled to 1% + SDXL + similarity filter         | 0.815         | N/A                    | 0.075 |
| Cat downsampled to 1% + SDXL + LoRA                      | 0.819         | N/A                    | 0.117 |
| Cat downsampled to 1% + class weighting                  | 0.766         | 0.764                  | 0.041 |
| Cat downsampled to 1% + label smoothing                  | 0.808         | 0.807                  | 0.000 |
| Downsampled + more augmentations                         | 0.150         | 0.148                  | 0.000 |
| Downsampled + random-oversample                          | 0.812         | 0.812                  | 0.008 |
| Downsampled + random-oversample + more augmentations     | 0.147         | 0.148                  | 0.000 |
| Downsampled + random-undersample                         | 0.217         | 0.217                  | 0.119 |
| Downsampled + SMOTE                                      | 0.811         | 0.809                  | 0.012 |
| FLUX1.dev                                                | 0.815         | 0.813                  | 0.039 |
| flux1.schnell                                            | 0.810         | 0.808                  | 0.022 |
| Full CIFAR10                                             | 0.868         | N/A                    | 0.748 |
| Full CIFAR10 + Full Synthetic                            | 0.871         | N/A                    | 0.741 |
| Full data + more augmentations                           | 0.105         | 0.105                  | 0.006 |
| stable-diffusion-3.5-large-turbo                         | 0.817         | 0.817                  | 0.094 |



## Data generation

Poniżej przedstawiono metody i skrypty użyte do generowania oraz rozszerzania zbioru danych o obrazy.

### 1. Generowanie obrazów za pomocą BeautifulPrompt (SDXL-Turbo)
Ta metoda wykorzystywała początkową strategię generowania promptów opartą na podejściu BeautifulPrompt (BeautifulPrompt: Towards Automatic Prompt Engineering for Text-to-Image Synthesis). Skupiała się ona na tworzeniu złożonych promptów dla konkretnych klas (np. kotów) w celu wygenerowania realistycznych obrazów.

- **Przykładowa struktura promptu:**
```
a photo taken from behind of a American Wirehair cat next to the cabinet, looking to the right. The cat has realistic fur textures, intricate details, and sharp features, with soft lighting and a clear focus. The image has a shallow depth of field, emphasizing the cat in fine detail. 8k, cinematic, photorealistic
```
- Wykorzystano różne zmienne, takie jak: `angle` (np. "a photo taken from above"), `breed` (100 ras, np. British Longhair), `preposition` (6, np. "on"), `furniture` (61 obiektów, np. "sofa"), `gaze` (12 kierunków spojrzenia, np. "looking up").
- Zastosowano do modelu `stable-diffusion-3.5-large-turbo`.

### 2. Generowanie obrazów w stylu SYNAuG (SDXL-Turbo)
Drugie podejście do generowania danych syntetycznych dla klasy `cat` wykorzystywało uproszczoną strukturę promptów, inspirowaną stylem "modyfikator + klasa" zaczerpniętym z badań SYNAuG.

- **Struktura promptu:** `"a photo of {selected_modifier} cat"`
- **Modyfikatory:** Losowo wybrane z listy 22 określeń (np. "realistic", "detailed", "photorealistic", "studio quality").
- **Liczba wygenerowanych obrazów:** 7,000 obrazów kotów.
- **Model:** `stabilityai/sdxl-turbo`.

### 3. Generowanie rozszerzonych obrazów CIFAR-10 (SDXL-Turbo z rozbudowanymi konceptami)
Najbardziej kompleksowa strategia generowania obrazów objęła wszystkie 10 klas CIFAR-10, wykorzystując rozbudowane listy deskryptorów i kontekstów, aby stworzyć bardzo zróżnicowany zbiór danych syntetycznych.

- **Struktura promptu:** `"{selected_quality} photo of a {selected_descriptor} {class_name} {selected_context}"`
- **Rozbudowane listy:**
    - `descriptor_type`: Specyficzne typy dla każdej klasy (np. "jet airplane" dla `airplane`, "sedan car" dla `automobile`, "tabby cat" dla `cat`).
    - `context_action`: Specyficzne akcje/konteksty dla każdej klasy (np. "flying in the clear blue sky" dla `airplane`, "driving on a multi-lane highway" dla `automobile`, "sitting comfortably on a plush sofa" dla `cat`).
    - `quality_modifiers`: Ogólne modyfikatory jakości (np. "photorealistic", "high resolution", "cinematic lighting").
- **Liczba docelowa obrazów:** 100,000 obrazów na każdą klasę.
- **Model:** `stabilityai/sdxl-turbo`.

### 4. Augmentacja Flux Redux
Zastosowano również metodę rozszerzania danych poprzez specyficzną augmentację Flux Redux, która przekształca istniejące obrazy wejściowe, generując ich syntetyczne warianty.

- **Wejście:** 50 oryginalnych obrazów (np. kotów).
- **Proces:** Każdy obraz jest przepuszczany przez pipeline Flux Redux (`black-forest-labs/FLUX.1-Redux-dev` jako prior i `black-forest-labs/FLUX.1-dev` jako główny pipeline), z dynamicznie zmienianym seedem dla każdej augmentacji.
- **Liczba wygenerowanych augmentacji:** 99 augmentacji na obraz, co dało łącznie około 4950 nowych przykładów.

### Wykorzystane zasoby i czasy generowania

**Wybrane modele do ewaluacji i generowania (z przybliżonymi czasami generowania 6,000 zdjęć):**
- `stable-diffusion-3.5-large-turbo` - 1.5 godziny
- `black-forest-labs/FLUX.1-schnell` - 2.5 godziny
- `black-forest-labs/FLUX.1-dev` - 14.5 godziny
- `SDXL Turbo` - TBD
- `FLUX Redux` - TBD

Wszystkie operacje były wykonywane na maszynie Lambda z konfiguracją: NVIDIA A100 (40GB VRAM), 8x vCPU, 1TB SSD, 1Gbps.