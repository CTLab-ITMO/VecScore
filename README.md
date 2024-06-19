# VecScore
Оценка качества векторных изображений (SVG-формат) методами глубокого обучения / Vector SVG-images quality assessment using deep learning


### Задачи (расположены поэтапно):
1) Сбор векторных изображений – parsers/main.py
2) Переименование и конвертация svg изображений в png расширение -- converting.py.
3) Некая статистика по разрешениям изобаржений и дальнейшая нормализация разрешений -- stata.py
4) Отчистка от дубликатов и нежелательных изображений -- clearing.py
5) Взятие эмбеддингов -- clipBlip/clipModel
6) Кластеризация -- clustering_models.py
7) Взятие описаний к изображениям -- clipBlip/blipModel

### Остальные файлы:
_paths.txt_ - для прописывания путей

_save_read.py_ - для храненения таких сущностей, как словари с очищенными/неочищенными эмбеддингами, получившиеся резульаты  кластеризации и т.д.

_visualization.py_ - функции для визуализации некоторых результатах.


### Сборка пайплайна:

#### Установка зависимостей:
```bash
conda create -n "nr_iqa" python=3.9
pip install -r base_requirements.txt
```

1) Прогоняете тетрадку 1_parse_data
2) Запускаете скрипт 2_load_data.py

Теперь ваши данные загружены в папку: toloka_parsed_data
Также по пути: "dataframes/loaded_parsed_toloka_dataset.csv" - у вас теперь есть датасет вот такой структуры:

```
	title	photo_url	res1	res2	res3	res4	mos1	mos2	mos3	mos4	final_mos	img_path
0	a black and white drawing of a man sitting o...	https://storage.yandexcloud.net/svg-images-fir...	1.0	1.0	1.0	1.0	1.0	1.0	1.0	1.0	1.000	/mnt/nvme0/jupyter-kazancev.danil7@wb-2ede4/pr...
1	a banana, lemon and a blueberry are shown	https://storage.yandexcloud.net/svg-images-fir...	5.0	2.0	1.0	1.0	5.0	3.5	3.0	3.0	3.625	/mnt/nvme0/jupyter-kazancev.danil7@wb-2ede4/pr...
```
