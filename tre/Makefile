PYTHON ?= python3
TRE_PYTHONPATH := $(CURDIR)/common:$(CURDIR)/deploy:$(CURDIR)/controller:$(CURDIR)/controller/tests:$(CURDIR)/service-manager:$(CURDIR)/calibration:$(CURDIR)/replayer:$(CURDIR)/loadgen_v1:$(CURDIR)/reissue:$(CURDIR)/ui:$(PYTHONPATH)

.PHONY: check check-redis smoke manifests

check:
	PYTHONPATH=$(TRE_PYTHONPATH) $(PYTHON) -m pytest -q common/tests deploy/tests controller/tests service-manager/tests calibration/tests replayer/tests loadgen_v1/tests reissue/tests ui/tests

# Real-Redis Lua tests (sleep reservations, fair writer lock) against a throwaway
# Redis container that is removed afterwards. `make check` stays hermetic: it runs
# the same tests against Python models of the scripts. Needs docker; override
# REDIS_TEST_IMAGE / REDIS_TEST_PORT / REDIS_TEST_CONTAINER if needed.
REDIS_TEST_IMAGE ?= redis:7.2-alpine
REDIS_TEST_PORT ?= 16399
REDIS_TEST_CONTAINER ?= tre-check-redis
REDIS_TESTS := service-manager/tests/test_sleep_reservations.py service-manager/tests/test_operations.py

check-redis:
	@docker rm -f $(REDIS_TEST_CONTAINER) >/dev/null 2>&1 || true
	docker run -d --rm --name $(REDIS_TEST_CONTAINER) -p 127.0.0.1:$(REDIS_TEST_PORT):6379 $(REDIS_TEST_IMAGE) >/dev/null
	@status=0; \
	for i in 1 2 3 4 5 6 7 8 9 10; do docker exec $(REDIS_TEST_CONTAINER) redis-cli ping >/dev/null 2>&1 && break; sleep 0.5; done; \
	TRE_TEST_REDIS_URL=redis://127.0.0.1:$(REDIS_TEST_PORT)/0 PYTHONPATH=$(TRE_PYTHONPATH) $(PYTHON) -m pytest -q -rA -k real-redis $(REDIS_TESTS) || status=$$?; \
	docker rm -f $(REDIS_TEST_CONTAINER) >/dev/null 2>&1; \
	exit $$status

smoke:
	PYTHONPATH=$(TRE_PYTHONPATH) $(PYTHON) deploy/registry_smoke.py --registry deploy/registry.yaml

manifests:
	PYTHONPATH=$(TRE_PYTHONPATH) $(PYTHON) deploy/gen_model_manifests.py --registry deploy/registry.yaml --output-dir deploy/models
