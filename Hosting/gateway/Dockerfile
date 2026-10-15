FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
      curl ca-certificates git libtinfo6 \
    && rm -rf /var/lib/apt/lists/*

ARG J_URL
RUN test -n "$J_URL" && mkdir -p /opt/j \
 && curl -fsSL "$J_URL" | tar -xz -C /opt/j --strip-components=1

RUN git clone --depth 1 https://github.com/jsoftware/convert_json /opt/j/addons/convert/json

WORKDIR /app
COPY citibike_api.ijs start.ijs ./
EXPOSE 8080
CMD ["/opt/j/bin/jconsole", "/app/start.ijs"]
