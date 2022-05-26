ifndef _BASE_MK_
_BASE_MK_ := 1

BASE_MK_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
BASE_MK_DIR := $(patsubst %/,%,$(dir $(BASE_MK_PATH)))
ROOT_DIR := $(abspath $(BASE_MK_DIR)/..)

## common

SHELL := /bin/bash
ECHO ?= echo -e
FIND ?= find

EMPTY :=
SPACE := $(EMPTY) $(EMPTY)
COMMA := ,
COLON := :
SEMICOLON := ;
QUOTE := "
SINGLE_QUOTE := '
OPEN_PAREN := (
CLOSE_PAREN := )

HOST_OS	:= $(shell uname -s 2>/dev/null | tr "[:upper:]" "[:lower:]")
HOST_ARCH := $(shell uname -m)

define echo
	text="$1"; options="$2"; \
	[ -z "$$options" ] && options="1;33"; \
	$(ECHO) "\033[$${options}m$${text}\033[0m"
endef

define md
	[ -e "$1" ] || (mkdir -p "$1" && $(call echo,MD: $1,33))
endef

define rm
	[ ! -h "$1" ] && [ ! -e "$1" ] || (rm -r "$1" && $(call echo,RM: $1,33))
endef

define rm_f
	dir="$2"; [ -e "$${dir}" ] || dir="."; \
	$(FIND) "$${dir}" -name "$1" | while read -r p; do \
		$(call rm,$$p); \
	done
endef

define rm_l
	for f in "$1"; do \
		$(call rm,$$f); \
	done
endef

## build

# args
#  release: release, otherwise debug
args ?= release

CC ?= cc
CCEXT ?= .c
CCFLAGS ?= -std=c99

CXX ?= c++
CXXEXT ?= .cc
CXXFLAGS ?= -std=c++14

LDFLAGS ?= -lm -ldl -lpthread

ifneq ($(filter release,$(args)),)
  BUILD_TYPE := release
  CCFLAGS += -O2 -DNDEBUG
  CXXFLAGS += -O2 -DNDEBUG
else
  BUILD_TYPE := debug
  CXXFLAGS += -g -DDEBUG
  CCFLAGS += -g -DDEBUG
endif
CCFLAGS += -Wall -Wextra -Wpedantic -fPIC
CXXFLAGS += -Wall -Wextra -Wpedantic -fPIC

LEX := flex
LEXEXT ?= .l
LEXEXTOUT ?= .yy$(CCEXT)

YACC := bison
YACCEXT ?= .y
YACCEXTOUT ?= .tab$(CCEXT)

PRINT_EXEC ?= 0
ifeq ($(PRINT_EXEC),1)
  EXEC ?= @echo "[@]"
endif

# rule

TARGET ?=
INCS ?=
LIBS ?=
SRCS ?=
OUTDIR ?= $(ROOT_DIR)/_build

## target

.PHONY: base_print
base_print:
	@$(call echo,Make $@)
	@echo BASE_MK_PATH: $(BASE_MK_PATH)
	@echo BASE_MK_DIR: $(BASE_MK_DIR)
	@echo ROOT_DIR: $(ROOT_DIR)
	@echo
	@echo HOST_OS: $(HOST_OS)
	@echo HOST_ARCH: $(HOST_ARCH)
	@echo
	@echo BUILD_TYPE: $(BUILD_TYPE)
	@echo CC: $(CC)
	@echo CCEXT: $(CCEXT)
	@echo CCFLAGS: $(CCFLAGS)
	@echo CXX: $(CXX)
	@echo CXXEXT: $(CXXEXT)
	@echo CXXFLAGS: $(CXXFLAGS)
	@echo LDFLAGS: $(LDFLAGS)
	@echo LEX: $(LEX)
	@echo LEXEXT: $(LEXEXT)
	@echo LEXEXTOUT: $(LEXEXTOUT)
	@echo YACC: $(YACC)
	@echo YACCEXT: $(YACCEXT)
	@echo YACCEXTOUT: $(YACCEXTOUT)
	@echo PRINT_EXEC: $(PRINT_EXEC)

endif # _BASE_MK_
