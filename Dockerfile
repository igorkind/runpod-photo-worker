# Используем проверенный образ PyTorch 2.2 + CUDA 12.1
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# 1. СИСТЕМНЫЕ ЗАВИСИМОСТИ (Кэшируются надолго)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. PYTHON ЗАВИСИМОСТИ (Отдельный слой)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade --force-reinstall -r requirements.txt

# 3. СКАЧИВАНИЕ МОДЕЛЕЙ (Самый тяжелый слой ~6GB)
# Копируем только скрипт загрузки, чтобы этот слой не зависел от изменений в handler.py
COPY builder.py .
RUN python builder.py && rm -rf /root/.cache/pip

# 4. КОД ВОРКЕРА (Меняется часто, легкий слой)
COPY handler.py .

# Запуск (без буферизации вывода)
CMD [ "python", "-u", "handler.py" ]