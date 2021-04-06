# I do modelling

This repo holds one of my Data Science Challenge Case work.

## Dependencies

* Install [poetry](https://github.com/python-poetry/poetry)
* Run `poetry install`

## Credentials

Credentials for BigQuery are expected to be present at
`/data/secrets/gcp-cross-sales/auth`. If you get a encryped file, run `gpg -d ~/Desktop/the_encryped_file.txt`, follow instructions and save in `/data/secrets/gcp-cross-sales/auth`.

## Run locally

* Running script for model: `poetry run python SEB_project/seb_project/scripts/script.py`
* Running Jupyter notebook: `poetry run jupyter notebook`

## Presentation 

* `SEB_project/seb_project/results/presentation.ppt`

# Docker Image

## Environment Image can be load from dockerhub alternatively 

* Command: `docker pull`