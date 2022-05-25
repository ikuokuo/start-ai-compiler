MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
CURR_DIR := $(patsubst %/,%,$(dir $(MKFILE_PATH)))
ROOT_DIR := $(abspath $(CURR_DIR)/../..)

include $(ROOT_DIR)/build/base.mk

TARGET := $(notdir $(CURR_DIR))

INCS +=
SRCS += $(CURR_DIR)/wc.l
LIBS +=

include $(ROOT_DIR)/build/rule.mk
