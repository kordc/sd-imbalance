import os
from PIL import Image, ImageFont, ImageDraw
import random

# --- Konfiguracja ścieżek i parametrów ---
BASE_MODEL_DIR = "model_comparison_images/"
CATS_DIR = "generated_data/cats_chosen/"
OUTPUT_GRID_FILENAME = "image_grid_enhanced.png"  # Zmieniona nazwa pliku wyjściowego, by nie nadpisać poprzedniego

# Docelowy rozmiar każdego obrazka w siatce (szerokość, wysokość)
# Obrazki zostaną przeskalowane do tego rozmiaru.
IMAGE_SIZE = (256, 256)

# Liczba kolumn w siatce (zgodnie z Twoim opisem, w każdym folderze jest 5 zdjęć)
GRID_COLS = 5

# Liczba wierszy dla zdjęć z modeli
GRID_ROWS_MODELS = 6

# Liczba wierszy dla zdjęć z kotów
GRID_ROWS_CATS = 1

# Całkowita liczba wierszy
TOTAL_GRID_ROWS = GRID_ROWS_MODELS + GRID_ROWS_CATS

# Odstęp między zdjęciami w pikselach
PADDING = 10

# Margines wokół całej siatki (top, right, bottom)
REGULAR_BORDER_PADDING = 20

# Rozmiar czcionki dla etykiet wierszy
FONT_SIZE = 36  # ZNACZNIE zwiększono rozmiar czcionki

# Kolejność folderów z modelami, zgodna z podanym 'ls -r'
MODEL_FOLDERS_ORDER = [
    "Stable_Diffusion_XL_Turbo",
    "Stable_Diffusion_XL",
    "Stable_Diffusion_3_Medium",
    "Stable_Diffusion_3.5_Large_Turbo",
    "FLUX1_schnell",
    "FLUX1_dev",
]

# --- Dynamiczne obliczanie lewego marginesu na podstawie szerokości tekstu ---
# Spróbuj załadować czcionkę do obliczeń szerokości tekstu.
# Możesz podać ścieżkę do czcionki systemowej, np. "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf" na Linuxie
try:
    font_path = "arial.ttf"  # Typowe dla Windows
    if not os.path.exists(font_path):
        # Przykładowe ścieżki dla Linuxa/macOS - dostosuj w razie potrzeby
        if os.path.exists("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        elif os.path.exists(
            "/System/Library/Fonts/Supplemental/Arial.ttf"
        ):  # macOS path
            font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
        elif os.path.exists("/usr/share/fonts/truetype/freefont/FreeSans.ttf"):
            font_path = "/usr/share/fonts/truetype/freefont/FreeSans.ttf"
        # Możesz dodać więcej ścieżek czcionek tutaj (np. poszukaj 'ttf' w /usr/share/fonts)

    if os.path.exists(font_path):
        font_for_measurement = ImageFont.truetype(font_path, FONT_SIZE)
    else:
        # Ostateczne wyjście awaryjne
        font_for_measurement = ImageFont.load_default()
        print(
            f"Ostrzeżenie: Nie znaleziono czcionki '{font_path}' (lub innych prób). Użyto domyślnej, co może wpłynąć na wygląd."
        )
except Exception as e:
    # Obsługa błędów podczas ładowania czcionki
    font_for_measurement = ImageFont.load_default()
    print(
        f"Błąd podczas ładowania czcionki: {e}. Użyto domyślnej, co może wpłynąć na wygląd."
    )


# Znajdź najdłuższą nazwę etykiety (po konwersji, np. "Stable Diffusion 3.5 Large Turbo")
all_labels = [name.replace("_", " ") for name in MODEL_FOLDERS_ORDER] + ["Random Cats"]
longest_label_text = max(all_labels, key=len)

# Utwórz tymczasowy obiekt ImageDraw do pomiaru szerokości tekstu
# Potrzebujemy kontekstu 'draw' do użycia .textbbox()
temp_img_for_measurement = Image.new("RGB", (1, 1))  # Minimalny obraz
temp_draw_for_measurement = ImageDraw.Draw(temp_img_for_measurement)

# Oblicz szerokość najdłuższego tekstu przy użyciu wybranej czcionki
# textbbox zwraca (left, top, right, bottom)
bbox = temp_draw_for_measurement.textbbox(
    (0, 0), longest_label_text, font=font_for_measurement
)
max_label_width = bbox[2] - bbox[0]  # Szerokość = prawo - lewo

# Ustaw lewy margines na podstawie szerokości najdłuższego tekstu + zapas
# Dodatkowy zapas 40 pikseli powinien zapewnić, że tekst nie będzie obcięty
LEFT_BORDER_PADDING = max_label_width + 40

# --- Koniec obliczeń dynamicznych ---


# --- Funkcja pomocnicza do ładowania i skalowania obrazów ---
def load_and_resize_image(
    image_path, target_size, fill_color=(200, 200, 200), error_color=(255, 0, 0)
):
    """Wczytuje obraz, skaluje go i obsługuje błędy."""
    if not os.path.exists(image_path):
        print(f"Błąd: Nie znaleziono obrazu: {image_path}. Używam szarego zastępczego.")
        return Image.new(
            "RGB", target_size, color=fill_color
        )  # Szary obrazek zastępczy

    try:
        img = Image.open(image_path).convert(
            "RGB"
        )  # Upewnij się, że obraz jest w trybie RGB
        img = img.resize(target_size, Image.LANCZOS)  # Skaluj z wysoką jakością
        return img
    except Exception as e:
        print(
            f"Błąd podczas ładowania/przetwarzania obrazu {image_path}: {e}. Używam czerwonego zastępczego."
        )
        return Image.new(
            "RGB", target_size, color=error_color
        )  # Czerwony obrazek zastępczy dla błędów


# --- 1. Zbieranie ścieżek do obrazów ---
all_image_paths_for_grid = []

# Obrazy z folderów modeli
for folder_name in MODEL_FOLDERS_ORDER:
    folder_path = os.path.join(BASE_MODEL_DIR, folder_name)

    if not os.path.isdir(folder_path):
        print(
            f"Ostrzeżenie: Nie znaleziono katalogu: {folder_path}. Pomijam ten wiersz."
        )
        # Dodaj pusty wiersz (lub wiersz z placeholderami), aby zachować strukturę siatki
        all_image_paths_for_grid.append([None] * GRID_COLS)
        continue

    # Filtruj tylko pliki obrazów (png, jpg, jpeg) i sortuj je alfabetycznie
    images_in_folder = sorted(
        [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )

    if len(images_in_folder) < GRID_COLS:
        print(
            f"Ostrzeżenie: Folder '{folder_name}' zawiera tylko {len(images_in_folder)} zdjęć, oczekiwano {GRID_COLS}. Uzupełniam brakujące szarymi placeholderami."
        )
        # Uzupełnij listę brakującymi obrazami placeholderami (None)
        images_in_folder.extend([None] * (GRID_COLS - len(images_in_folder)))
    elif len(images_in_folder) > GRID_COLS:
        print(
            f"Ostrzeżenie: Folder '{folder_name}' zawiera {len(images_in_folder)} zdjęć, oczekiwano {GRID_COLS}. Biorę tylko pierwsze {GRID_COLS} zdjęć."
        )
        images_in_folder = images_in_folder[:GRID_COLS]

    all_image_paths_for_grid.append(images_in_folder)

# Obrazy z kotów (ostatni wiersz)
cat_image_paths = []
if not os.path.isdir(CATS_DIR):
    print(
        f"Błąd: Katalog z kotami nie znaleziony: {CATS_DIR}. Nie mogę dodać zdjęć kotów."
    )
    cat_image_paths = [None] * GRID_COLS  # Wiersz z placeholderami
else:
    all_cat_files = [
        os.path.join(CATS_DIR, f)
        for f in os.listdir(CATS_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    if len(all_cat_files) < GRID_COLS:
        print(
            f"Ostrzeżenie: Za mało zdjęć kotów ({len(all_cat_files)}), oczekiwano {GRID_COLS}. Uzupełniam brakujące szarymi placeholderami."
        )
        cat_image_paths = all_cat_files + [None] * (GRID_COLS - len(all_cat_files))
    else:
        cat_image_paths = random.sample(
            all_cat_files, GRID_COLS
        )  # Wybierz losowo 5 zdjęć

all_image_paths_for_grid.append(cat_image_paths)

# --- 2. Ładowanie i skalowanie obrazów ---
loaded_images_grid = []
for row_paths in all_image_paths_for_grid:
    row_images = []
    for img_path in row_paths:
        if img_path:
            row_images.append(load_and_resize_image(img_path, IMAGE_SIZE))
        else:
            # Użyj szarego placeholderu dla brakujących obrazów (np. z folderów z zbyt małą liczbą zdjęć)
            row_images.append(Image.new("RGB", IMAGE_SIZE, color=(150, 150, 150)))
    loaded_images_grid.append(row_images)

# --- 3. Tworzenie siatki obrazów ---
# Obliczanie wymiarów końcowego obrazu siatki
grid_width = GRID_COLS * IMAGE_SIZE[0] + (GRID_COLS - 1) * PADDING
grid_height = TOTAL_GRID_ROWS * IMAGE_SIZE[1] + (TOTAL_GRID_ROWS - 1) * PADDING

# Obliczenie pełnych wymiarów obrazu, uwzględniając różne marginesy
final_grid_width = LEFT_BORDER_PADDING + grid_width + REGULAR_BORDER_PADDING
final_grid_height = (
    REGULAR_BORDER_PADDING + grid_height + REGULAR_BORDER_PADDING
)  # Margines górny + dolny

# Utwórz nowy, pusty obraz (tło białe)
grid_image = Image.new("RGB", (final_grid_width, final_grid_height), color="white")
draw = ImageDraw.Draw(grid_image)

# Załaduj czcionkę do rysowania (ta sama, której użyto do pomiaru)
font = font_for_measurement

# Wklejanie obrazów do siatki i dodawanie etykiet
for r_idx, row_images in enumerate(loaded_images_grid):
    # Wylicz pozycję Y dla bieżącego wiersza obrazków (uwzględnia górny margines)
    y_offset_for_row_content = REGULAR_BORDER_PADDING + r_idx * (
        IMAGE_SIZE[1] + PADDING
    )

    # Przygotowanie tekstu etykiety
    label_text = ""
    if r_idx < GRID_ROWS_MODELS:
        label_text = MODEL_FOLDERS_ORDER[r_idx].replace("_", " ")
    else:
        label_text = "Random Cats"

    # Narysuj tekst na lewym marginesie
    # X: Środek lewego marginesu (horizontal center of the padding area)
    text_x = LEFT_BORDER_PADDING / 2
    # Y: Środek wiersza obrazków (vertical center of the image row)
    text_y = y_offset_for_row_content + IMAGE_SIZE[1] / 2

    # Użyj anchor="mm" (middle-middle) aby punkt (text_x, text_y) był środkiem bounding boxu tekstu
    # To zapewnia, że tekst jest wyśrodkowany zarówno poziomo w marginesie, jak i pionowo w rzędzie obrazków.
    draw.text((text_x, text_y), label_text, font=font, fill=(0, 0, 0), anchor="mm")

    # Wklejanie obrazów do siatki
    for c_idx, img in enumerate(row_images):
        # X: Pozycja zaczyna się od LEFT_BORDER_PADDING
        x_offset_for_image = LEFT_BORDER_PADDING + c_idx * (IMAGE_SIZE[0] + PADDING)

        # Wklej obraz
        grid_image.paste(img, (x_offset_for_image, y_offset_for_row_content))

# --- 4. Zapisywanie siatki obrazów ---
try:
    grid_image.save(OUTPUT_GRID_FILENAME)
    print(f"\nSiatka obrazów została pomyślnie zapisana jako: {OUTPUT_GRID_FILENAME}")
except Exception as e:
    print(f"Błąd podczas zapisywania obrazu siatki: {e}")
