PYTHON ?= python3
TRE_PYTHONPATH := $(CURDIR)/common:$(CURDIR)/deploy:$(CURDIR)/controller:$(CURDIR)/controller/tests:$(CURDIR)/service-manager:$(CURDIR)/calibration:$(CURDIR)/replayer:$(CURDIR)/loadgen_v1:$(CURDIR)/reissue:$(CURDIR)/ui:$(PYTHONPATH)

.PHONY: check smoke manifests

check:
	PYTHONPATH=$(TRE_PYTHONPATH) $(PYTHON) -m pytest -q common/tests deploy/tests controller/tests service-manager/tests calibration/tests replayer/tests loadgen_v1/tests reissue/tests ui/tests

smoke:
	PYTHONPATH=$(TRE_PYTHONPATH) $(PYTHON) deploy/registry_smoke.py --registry deploy/registry.yaml

manifests:
	PYTHONPATH=$(TRE_PYTHONPATH) $(PYTHON) deploy/gen_model_manifests.py --registry deploy/registry.yaml --output-dir deploy/models
