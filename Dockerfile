# Используем PyTorch 2.2 + CUDA 12.1
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# 1. Системные зависимости
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Python зависимости
COPY requirements.txt .

# 🔥 ВАЖНО: Сначала удаляем предустановленные конфликтующие библиотеки
RUN pip uninstall -y diffusers transformers accelerate huggingface_hub || true

# 🔥 Устанавливаем наши версии начисто
RUN pip install --no-cache-dir --upgrade --force-reinstall -r requirements.txt

# 3. Копируем код
COPY builder.py .
COPY handler.py .

# 4. Скачивание моделей (кэш pip удаляем, чтобы уменьшить размер)
RUN python builder.py && rm -rf /root/.cache/pip

CMD [ "python", "-u", "handler.py" ]