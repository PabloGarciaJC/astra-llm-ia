#!/bin/bash
source "$(dirname "$0")/_header.sh"
print_header "llm-trainer"
docker compose run --rm llm-trainer bash
