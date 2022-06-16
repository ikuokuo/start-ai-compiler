MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
CURR_DIR := $(patsubst %/,%,$(dir $(MKFILE_PATH)))

include $(CURR_DIR)/build/base.mk

.DEFAULT_GOAL := release
.PHONY: release debug clean

release:
	@$(call echo,Make $@)
	$(CURR_DIR)/build.sh release

debug:
	@$(call echo,Make $@)
	$(CURR_DIR)/build.sh debug

clean:
	@$(call echo,Make $@)
	@$(call rm,$(OUTDIR))
	@$(call rm_f,*.lex.[h|c])
	@$(call rm_f,*.tab.[h|c])
	@$(call rm_f,*.yy.[h|c])
	@#$(CURR_DIR)/clean.sh
