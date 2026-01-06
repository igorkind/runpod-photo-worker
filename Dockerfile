# Используем PyTorch 2.2 + CUDA 12.1
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# 1. Системные зависимости
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Python зависимости (сначала копируем только requirements)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Код и Скрипты
COPY builder.py .
COPY handler.py .

# 4. Скачивание моделей
# Удаляем только кэш pip. Кэш моделей (huggingface) оставляем!
RUN python builder.py && rm -rf /root/.cache/pip

CMD [ "python", "-u", "handler.py" ]