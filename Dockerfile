# ИСПОЛЬЗУЕМ БОЛЕЕ НОВЫЙ ОБРАЗ (PyTorch 2.2 + CUDA 12.1)
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Системные зависимости
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY builder.py .
COPY handler.py .

# Запускаем builder и удаляем ТОЛЬКО кэш pip. Кэш моделей оставляем!
RUN python builder.py && \
    rm -rf /root/.cache/pip

CMD [ "python", "-u", "handler.py" ]