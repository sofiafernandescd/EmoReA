# Use uma imagem base leve
FROM python:3.11-slim

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Criar diretório de trabalho
WORKDIR /app

# Copiar dependências e instalar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o código do backend
COPY . .

# Variáveis de ambiente
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Expor a porta (para debug local)
EXPOSE 8080

# Comando de arranque (dinâmico)
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}
