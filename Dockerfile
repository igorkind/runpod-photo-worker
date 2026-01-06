# Используем PyTorch 2.2 + CUDA 12.1
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# 1. Слой системных библиотек (редко меняется)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Слой Python библиотек (меняется при правке requirements.txt)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Слой скриптов (часто меняется, весит мало)
COPY builder.py .
COPY handler.py .

# 4. Слой моделей (ОЧЕНЬ ТЯЖЕЛЫЙ ~6GB)
# Мы выносим его в конец. Если он уже был загружен, Docker его переиспользует.
# builder.py скачивает модели.
RUN python builder.py && rm -rf /root/.cache/pip

CMD [ "python", "-u", "handler.py" ]