FROM python:3.12-slim-bookworm
RUN pip install --no-cache-dir numpy==2.2.6 scipy==1.15.3 pandas==2.2.3
COPY src/geotech_rl /opt/geotech/geotech_rl
RUN pip install --no-cache-dir --no-deps /opt/geotech 2>/dev/null || true
ENV PYTHONPATH=/opt/geotech
WORKDIR /work
