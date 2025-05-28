# 1. Base on NVIDIA Isaac Sim runtime
ARG BASE_IMAGE=nvcr.io/nvidia/isaac-sim:4.5.0 
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-c"]
ENV ACCEPT_EULA=Y
ENV OMNI_KIT_ACCEPT_EULA=yes


# 2. Install Python 3.10 and extra tools (git, curl)
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
      python3.10 python3.10-venv python3-pip \
      git curl && \
    rm -rf /var/lib/apt/lists/*

# 3. Ensure python and pip map to Python 3.10
RUN ln -sf /usr/bin/python3.10 /usr/bin/python \
 && ln -sf /usr/bin/pip3   /usr/bin/pip

# 4. Install & upgrade pip, setuptools, then Astral UV
RUN pip install --upgrade pip setuptools \
 && pip install uv

# 5. Set up workspace and copy lockfiles for layer caching
ARG WORKDIR=/workspace/isaaclab
WORKDIR ${WORKDIR}
COPY pyproject.toml uv.lock ./

RUN mkdir -p /venv
ENV VIRTUAL_ENV=/venv
ENV UV_PROJECT_ENVIRONMENT=/venv

# 6. Copy all editable IsaacLab sources in one go
COPY IsaacLab/source ./IsaacLab/source

RUN uv venv /venv

# 7. Install dependencies from lockfile
ENV UV_LINK_MODE=copy
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project


# VOLUME ["${WORKDIR}/.venv"]
# ENV PATH="${WORKDIR}/.venv/bin:$PATH"

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# 8. (Optional) Expose any ports your code serves (e.g., a dashboard or API)
# EXPOSE 8888

# 9. Override isaacsim entrypoint
ENTRYPOINT ["/entrypoint.sh"]