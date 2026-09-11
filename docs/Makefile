# Minimal Sphinx Makefile
SPHINXOPTS    ?=
# Prefer the project venv's sphinx-build if it exists (has the right numpy/pandas)
_VENV_SPHINX := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))../.venv/bin/sphinx-build
SPHINXBUILD   ?= $(if $(wildcard $(_VENV_SPHINX)),$(_VENV_SPHINX),sphinx-build)
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: help clean html linkcheck Makefile

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf $(BUILDDIR)

html:
	$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

linkcheck:
	$(SPHINXBUILD) -M linkcheck "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
