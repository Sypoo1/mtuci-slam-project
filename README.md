# MTUCI SLAM Project

> ⚡️ Этот репозиторий основан на проекте [Monocular SLAM for Robotics implementation in python (learnopencv)](https://github.com/spmallick/learnopencv/tree/master/Monocular%20SLAM%20for%20Robotics%20implementation%20in%20python)

---

## Быстрый старт

### 1. Клонирование репозитория

```bash
git clone https://github.com/Sypoo1/mtuci-slam-project.git
cd mtuci-slam-project
```

### 2. Установка зависимостей

#### Системные зависимости (Ubuntu/Debian)

Перед установкой зависимостей рекомендуется обновить список пакетов и уже установленные пакеты:

```bash
sudo apt update && sudo apt upgrade -y
```

Установите необходимые библиотеки:

```bash
sudo apt-get install libglew-dev
sudo apt-get install cmake
sudo apt-get install ffmpeg libavcodec-dev libavutil-dev libavformat-dev libswscale-dev
sudo apt-get install libdc1394-22-dev libraw1394-dev
sudo apt-get install libjpeg-dev libpng-dev libtiff5-dev libopenexr-dev
```

#### Python-зависимости

Рекомендуется использовать виртуальное окружение:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Установите все необходимые Python-библиотеки одной командой:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---
## Установка Pangolin для Python

1. **Активируйте виртуальное окружение (если ещё не активировано):**
   ```bash
   source .venv/bin/activate
   ```
2. **Склонируйте репозиторий с Python-обёрткой Pangolin:**
   ```bash
   git clone https://github.com/uoip/pangolin.git
   cd pangolin
   ```
3. **Создайте папку для сборки и перейдите в неё:**
   ```bash
   mkdir build
   cd build
   ```
4. **Соберите проект с помощью CMake:**
   ```bash
   cmake .. -DPython_EXECUTABLE=$(which python3) -DBUILD_PANGOLIN_FFMPEG=OFF
   ```
5. **Скомпилируйте библиотеку:**
   ```bash
   make -j8
   cd ..
   ```
6. **Исправьте файл `setup.py`:**
   - Откройте файл `setup.py` в любом текстовом редакторе.
   - Найдите функцию `def run(self):` и добавьте строку:
     ```python
     install_dirs = [install_dir]
     ```
   - В итоге функция должна выглядеть так:
     ```python
     def run(self):
         install_dir = get_python_lib()
         install_dirs = [install_dir]
         # ... остальной код ...
     ```
7. **Установите библиотеку:**
   ```bash
   python setup.py install
   ```

> **Примечание:**
> Если при сборке (`make -j8`) или установке возникнут ошибки, обратитесь к [этому комментарию](https://github.com/uoip/pangolin/issues/33#issuecomment-717655495) и [этому решению](https://github.com/uoip/pangolin/issues/20#issuecomment-498211997) на GitHub.


---

## Настройка параметров камеры

Перед запуском SLAM-системы необходимо задать параметры вашей камеры в файле `main.py`:

- **Размер кадра:**
  В начале файла задайте разрешение вашей камеры:
  ```python
  W, H = 1280, 720  # Замените на разрешение вашей камеры
  ```
- **Калибровочная матрица камеры (K):**
  Для корректной работы SLAM требуется матрица внутренних параметров камеры. Получить её можно с помощью функции:
  ```python
  from calibration import get_camera_matrix
  K = get_camera_matrix()
  ```
  Функция автоматически загрузит матрицу из файла после калибровки (см. ниже).

---

## Калибровка камеры

Для точной работы SLAM необходимо откалибровать вашу камеру. Это позволит получить матрицу внутренних параметров (K) и коэффициенты дисторсии.

### Быстрая калибровка (одной командой)

1. Сделайте 15-20 фотографий шахматной доски с разных ракурсов и поместите их в папку `chessboard_images/` (поддерживаются форматы: .jpg, .png, .jpeg, .bmp).
2. Запустите калибровку:
   ```bash
   python -m calibration.camera_calibrator --find-size --debug --force --test
   ```
   Эта команда:
   - Автоматически определит оптимальный размер шахматной доски
   - Проведёт калибровку
   - Покажет результаты и сохранит матрицу камеры в файлы `calibration/camera_calibration.npz` и `camera_calibration.npz`

### Калибровка по шагам (для контроля)

1. **Подготовьте изображения**
   - Сделайте 15-20 фото шахматной доски с разных углов
   - Поместите их в папку `chessboard_images/`

2. **Определите размер доски**
   ```bash
   python -m calibration.camera_calibrator --find-size
   ```

3. **Выполните калибровку**
   (замените 7x6 на найденный размер)
   ```bash
   python -m calibration.camera_calibrator --size=7x6 --debug --force
   ```

4. **Проверьте результат**
   ```bash
   python -m calibration.camera_calibrator --test
   ```

#### Использование матрицы камеры в проекте

После калибровки используйте функцию:
```python
from calibration import get_camera_matrix
K = get_camera_matrix()
```
- `K` — матрица внутренних параметров камеры

Матрица будет автоматически использоваться в SLAM через вызов `get_camera_matrix()` в `main.py`.

#### Дополнительные параметры калибровки
- `--folder=path` — папка с изображениями (по умолчанию `chessboard_images`)
- `--output=path` — путь для сохранения калибровки (по умолчанию `calibration/camera_calibration.npz`)
- `--force` — принудительная перекалибровка
- `--debug` — отображение процесса поиска доски
- `--test` — просмотр результатов калибровки

#### Просмотр текущей матрицы камеры

Чтобы вывести информацию о текущей матрице камеры без калибровки:
```bash
python -m calibration.show_matrix
```

---

## Как работает программа

При запуске основного файла `main.py`:
- Откроется 3D-окно с визуализацией карты и траекторией движения камеры.
- В левом нижнем углу будет отображаться 2D-изображение с наложенными ключевыми точками и связями.
- Управление:
  - **Клавиша `p`** — поставить процесс отрисовки на паузу/снять с паузы.
  - **Клавиша `Tab`** — переключение в полноэкранный режим карты и обратно.
  - **Клавиша `Esc`** — завершить программу. При завершении карта автоматически сохранится в файл `map.npz`.
- Полученный файл `map.npz` можно затем просмотреть с помощью скрипта `view_map.py` (см. выше).

> ⚠️ **ВНИМАНИЕ:** При каждом новом запуске `main.py` файл карты `map.npz` будет ПЕРЕЗАПИСАН! Если вы хотите сохранить предыдущую карту, переименуйте или скопируйте файл до следующего запуска.

---

## Как запустить проект?

После установки всех зависимостей и настройки параметров камеры выполните:

```bash
python main.py
```

---

## Визуализация 3D-карты

Для отображения построенной 3D-карты, которая была сохранена после работы SLAM в файл `map.npz`, используйте скрипт `view_map.py`:

```bash
python view_map.py
```

Этот скрипт откроет интерактивное окно с 3D-визуализацией карты и траекторией камеры, полученной при запуске основного файла `main.py`.

---

## Структура проекта

```
.
├── calibration
│   ├── camera_calibration.npz
│   ├── camera_calibrator.py
│   ├── get_camera_matrix.py
│   ├── __init__.py
│   └── show_matrix.py
├── chessboard_images
├── extractor.py
├── main.py
├── map.npz
├── notebooks
│   ├── bundle_adjustment.ipynb
│   ├── mapping.ipynb
│   └── SLAM_pipeline_step_by_step.ipynb
├── pointmap.py
├── pyproject.toml
├── README.md
├── requirements.txt
├── test.ipynb
├── utils.py
├── uv.lock
├── videos
│   ├── car.mp4
│   ├── test_countryroad.mp4
│   ├── test_nyc.mp4
│   └── test_ohio.mp4
└── view_map.py
```

---

## Jupyter-ноутбуки

- **SLAM_pipeline_step_by_step.ipynb** — описание всего пайплайна SLAM.
- **mapping.ipynb** — пример по построению карты ([источник](https://github.com/SiddhantNadkarni/Parallel_SFM)).
- **bundle_adjustment.ipynb** — разбор bundle adjustment и g2o ([источник](https://github.com/maxcrous/multiview_notebooks)).

Для работы первого ноутбука потребуется датасет KITTI (grayscale, 22 ГБ):
[Скачать KITTI Odometry](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)

---

Если потребуется помощь или возникнут вопросы — создайте issue или обратитесь к авторам исходных репозиториев.
