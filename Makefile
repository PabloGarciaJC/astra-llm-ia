-include .env

# ==============================
# FIRST-TIME SETUP
# ==============================
init-app: prepare train web

# ==============================
# PREPARE
# ==============================
prepare:
	@[ ! -f .env ] && cp .env.example .env || true
	@mkdir -p llm-trainer/data llm-trainer/output
	@chmod +x scripts/*.sh

# ==============================
# DOCKER
# ==============================
up:
	docker compose up -d --build

down:
	docker compose down

restart:
	docker compose restart

build:
	docker compose build --no-cache

logs:
	docker compose logs -f

ps:
	docker compose ps

# ==============================
# ENTRENAMIENTO
# ==============================
train:
	docker compose run --rm llm-trainer python train.py

train-logs:
	docker compose logs -f llm-trainer

train-shell:
	docker compose run --rm llm-trainer bash

# ==============================
# GENERACIÓN
# ==============================
generate:
	docker compose run --rm -e PROMPT="$(PROMPT)" -e GEN_LEN="$(or $(GEN_LEN),300)" -e TEMPERATURE="$(or $(TEMPERATURE),0.8)" llm-trainer python generate.py

# ==============================
# INTERFAZ WEB
# ==============================
web:
	docker compose up -d --build llm-web
	@echo ""
	@echo "  Astra LLM AI corriendo en http://localhost:8080"
	@echo ""

web-stop:
	docker compose stop llm-web

web-logs:
	docker compose logs -f llm-web

# ==============================
# SYSTEM / DIAGNOSTIC
# ==============================
status:
	docker ps -a
	docker images
	docker volume ls
	docker network ls

# ==============================
# DESTRUCTIVE CLEANUP (DANGER)
# ==============================
nuke:
	@echo "PELIGRO: Esto borrara todos los contenedores, volumenes e imagenes de Docker."
	@echo "No hay vuelta atras."
	@read -p "Confirmas que quieres continuar? [s/N]: " confirm && \
	  [ "$$confirm" = "s" ] || [ "$$confirm" = "S" ] || (echo "Cancelado." && exit 1)
	docker compose down -v
	docker stop $$(docker ps -aq) 2>/dev/null || true
	docker rm $$(docker ps -aq) 2>/dev/null || true
	docker volume rm $$(docker volume ls -q) 2>/dev/null || true
	docker system prune -a --volumes -f
	@echo "Limpieza completada."

# ==============================
# SHELL ACCESS
# ==============================
bash-trainer:
	-./scripts/docker-bash-trainer.sh

# ==============================
# CLAUDE
# ==============================
github-feature:
	@read -p "Describe la nueva feature: " feature; \
	claude "Eres un asistente de desarrollo. El usuario quiere implementar esta feature: '$$feature'. Haz exactamente estos pasos en orden: 1) Usa el MCP de GitHub para crear un issue con título y descripción detallada de la feature. 2) Crea una rama local con el formato feature/<numero-issue>-<nombre-corto> y haz checkout a ella. 3) Implementa la solución completa en el código del proyecto. 4) Haz commit y push de los cambios a esa rama."
