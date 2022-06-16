MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
CURR_DIR := $(patsubst %/,%,$(dir $(MKFILE_PATH)))
ROOT_DIR := $(abspath $(CURR_DIR)/../..)

CCEXT := .c
LEXEXTOUT := .lex$(CCEXT)
include $(ROOT_DIR)/build/base.mk

TARGET := $(notdir $(CURR_DIR))
CCFLAGS += -Wno-implicit-fallthrough
SRCS += $(CURR_DIR)/purecalc.l $(CURR_DIR)/purecalc.y $(CURR_DIR)/purecalcfuncs.c

NO_DEPS := 1
include $(ROOT_DIR)/build/rule.mk

.PHONY: fixlexparam
fixlexparam: purecalc.tab.c
	@sed -i -E 's/(pp->)?(scaninfo\);)/pp->\2/g' $(CURR_DIR)/purecalc.tab.c

$(OBJDIR)/purecalcfuncs.c.o: purecalcfuncs.c purecalc.lex.h purecalc.tab.h
$(OBJDIR)/purecalc.lex.c.o: purecalc.tab.h purecalc.h
$(OBJDIR)/purecalc.tab.c.o: purecalc.lex.h purecalc.h fixlexparam
purecalc.lex.h: purecalc.lex.c
purecalc.tab.h: purecalc.tab.c
