# Stage 1: Build dependencies
FROM python:3.11.10-slim

# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /app

# Set environment variables for uv optimization
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Copy project metadata and lockfile
COPY requirements.txt service.py models pyproject.toml uv.lock ./

# Install dependencies using uv sync
# Use a cache mount for faster builds
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --no-install-project --no-dev

# Copy the rest of the application code
COPY . .

EXPOSE 8010

CMD ["uv", "run", "service.py"]