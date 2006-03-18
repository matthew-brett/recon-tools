python := /bin/env python
version := 0.3a
package_name := imaging-tools
rpm_name := $(package_name)-$(version)
checkinstall := /usr/sbin/checkinstall
docs_dir := $(shell pwd)/docs
test_dir := $(shell pwd)/test

# make a tools rpm (includes documentation) (requires rpmdevtools)
tools-rpm: docs
	$(python) setup.py bdist_rpm

# install the tools package and documentation
install: install-docs FORCE
	$(python) setup.py install

test: FORCE
	echo This Makefile does not currently run $(package_name) tests.

clean: docs-clean FORCE
	./clean

FORCE:

include $(docs_dir)/Makefile
