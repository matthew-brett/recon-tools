# This makefile must be included from the top-level imaging-tools makefile
# in order for its targets to be built since it depends on variables defined
# there
docs_scripts := $(docs_dir)/scripts
docs_dest := /usr/share/doc/$(package_name)-$(version)
docs_rpm_name := $(package_name)-docs-$(version)
api_dir := api
tool_usage := tool-usage.txt
python_source_files := $(shell find root/ -name "*.py")

docs: api ${tool_usage}

# make a documentation rpm (requires checkinstall# make a documentation rpm (requires checkinstall)
docs-rpm: docs
	$(checkinstall) -R \
      --pkgname=$(package_name)-docs \
      --pkgversion=$(version)\
      --showinstall=no --nodoc \
	  $(docs_scripts)/install-docs $(docs_dir) $(docs_dest)

# make tool usage file
${tool_usage}: ${python_source_files}
	cd $(docs_scripts) && ./tool-usage

# generage api docs
api: ${python_source_files}
	echo $(docs_scripts)
	cd $(docs_scripts) && ./run-epydoc

# install documentation
install-docs: docs
    echo Running docs/Makefile:install-docs
	$(docs_scripts)/install-docs $(docs_dir) $(docs_dest)

# remove all generated documentation
docs-clean: FORCE
	rm -rf $(api_dir)
	rm -f $(tool_usage)

FORCE:
