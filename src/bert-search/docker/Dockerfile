FROM python:3.8.7-slim
WORKDIR /code
RUN apt-get -y update && \
    apt-get -y install gcc && \
    apt-get -y install mecab libmecab-dev && \
    apt-get autoremove -y &&\
    apt-get clean
ENV LANG en_US.utf8
COPY pyproject.toml /code/
RUN pip install -U pip &&\
  pip install --no-cache-dir poetry && \
  poetry config virtualenvs.create false && \
  poetry install && \
  rm pyproject.toml
