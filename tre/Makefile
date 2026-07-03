PYTHON ?= python3
TRE_PYTHONPATH := $(CURDIR)/common:$(CURDIR)/deploy:$(CURDIR)/controller:$(CURDIR)/controller/tests:$(PYTHONPATH)

.PHONY: check smoke manifests

check:
	PYTHONPATH=$(TRE_PYTHONPATH) $(PYTHON) -m pytest -q common/tests deploy/tests controller/tests

smoke:
	PYTHONPATH=$(TRE_PYTHONPATH) $(PYTHON) -c "from tre_common.registry import load_registry; r=load_registry('deploy/registry.yaml'); errors=r.validate(); assert not errors, errors; print('tre smoke ok')"

manifests:
	PYTHONPATH=$(TRE_PYTHONPATH) $(PYTHON) deploy/gen_model_manifests.py --registry deploy/registry.yaml --output-dir deploy/models
