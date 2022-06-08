MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
CURR_DIR := $(patsubst %/,%,$(dir $(MKFILE_PATH)))
ROOT_DIR := $(abspath $(CURR_DIR)/../..)

include $(ROOT_DIR)/build/base.mk

TARGET := $(notdir $(CURR_DIR))
CCFLAGS += -Wno-implicit-fallthrough
SRCS += $(CURR_DIR)/calc.l $(CURR_DIR)/calc.y $(CURR_DIR)/calc.c

include $(ROOT_DIR)/build/rule.mk
