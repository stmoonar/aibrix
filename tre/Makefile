PYTHON ?= python3
TRE_PYTHONPATH := $(CURDIR)/common:$(CURDIR)/deploy:$(CURDIR)/controller:$(CURDIR)/controller/tests:$(CURDIR)/service-manager:$(CURDIR)/calibration:$(CURDIR)/replayer:$(CURDIR)/loadgen_v1:$(CURDIR)/reissue:$(CURDIR)/ui:$(PYTHONPATH)

.PHONY: check check-serial check-redis smoke manifests

# Test suites (directory names); `make check SUITES="controller service-manager"` runs a subset.
SUITES ?= common deploy controller service-manager calibration replayer loadgen_v1 reissue ui
CHECK_LOG_DIR ?= .check-logs

# Default: one pytest process per suite, all in parallel (no xdist needed). Per-suite
# log in $(CHECK_LOG_DIR)/<suite>.log; a summary table is printed at the end and the failing
# suites' pytest tails are echoed. Exit status is non-zero if any suite failed.
# -p no:cacheprovider: concurrent processes must not share/write .pytest_cache.
check:
	@mkdir -p $(CHECK_LOG_DIR); rm -f $(CHECK_LOG_DIR)/*.log $(CHECK_LOG_DIR)/*.rc $(CHECK_LOG_DIR)/*.sec; \
	t0=$$(date +%s); \
	for s in $(SUITES); do \
	  ( s0=$$(date +%s); \
	    PYTHONPATH=$(TRE_PYTHONPATH) $(PYTHON) -m pytest -q -p no:cacheprovider $$s/tests >$(CHECK_LOG_DIR)/$$s.log 2>&1; rc=$$?; \
	    echo $$rc >$(CHECK_LOG_DIR)/$$s.rc; echo $$(( $$(date +%s) - s0 )) >$(CHECK_LOG_DIR)/$$s.sec ) & \
	done; wait; \
	status=0; printf '\n%-16s %-7s %s\n' SUITE RESULT 'TIME(s)  SUMMARY'; \
	for s in $(SUITES); do \
	  rc=$$(cat $(CHECK_LOG_DIR)/$$s.rc 2>/dev/null || echo 99); \
	  [ "$$rc" = 0 ] || status=1; \
	  printf '%-16s %-7s %-8s %s\n' $$s $$([ "$$rc" = 0 ] && echo ok || echo FAIL) "$$(cat $(CHECK_LOG_DIR)/$$s.sec 2>/dev/null)" "$$(tail -n1 $(CHECK_LOG_DIR)/$$s.log)"; \
	done; \
	for s in $(SUITES); do \
	  rc=$$(cat $(CHECK_LOG_DIR)/$$s.rc 2>/dev/null || echo 99); \
	  [ "$$rc" = 0 ] || { echo; echo "===== FAILED: $$s (rc=$$rc, full log: $(CHECK_LOG_DIR)/$$s.log) ====="; tail -n 40 $(CHECK_LOG_DIR)/$$s.log; }; \
	done; \
	echo; echo "wall time: $$(( $$(date +%s) - t0 ))s"; exit $$status

# Original serial behaviour (single pytest process over all suites).
check-serial:
	PYTHONPATH=$(TRE_PYTHONPATH) $(PYTHON) -m pytest -q common/tests deploy/tests controller/tests service-manager/tests calibration/tests replayer/tests loadgen_v1/tests reissue/tests ui/tests

# Real-Redis Lua tests (sleep reservations, fair writer lock, maintenance lock) against a throwaway
# Redis container that is removed afterwards. `make check` stays hermetic: it runs
# the same tests against Python models of the scripts. Needs docker; override
# REDIS_TEST_IMAGE / REDIS_TEST_PORT / REDIS_TEST_CONTAINER if needed.
REDIS_TEST_IMAGE ?= redis:7.2-alpine
REDIS_TEST_PORT ?= 16399
REDIS_TEST_CONTAINER ?= tre-check-redis
REDIS_TESTS := service-manager/tests/test_sleep_reservations.py service-manager/tests/test_operations.py service-manager/tests/test_maintenance_lock.py

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
