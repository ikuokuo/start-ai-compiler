.DEFAULT_GOAL := bin

ifeq ($(strip $(TARGET)),)
  $(error ERROR$(COLON) TARGET not given)
endif
ifeq ($(strip $(SRCS)),)
  $(error ERROR$(COLON) SRCS not given)
endif

TRTDIR := $(OUTDIR)/$(HOST_OS)-$(HOST_ARCH)/$(BUILD_TYPE)/$(TARGET)
BINDIR := $(TRTDIR)/bin
OBJDIR := $(TRTDIR)/obj

SRCS_CC := $(filter %$(CCEXT),$(SRCS))
SRCS_CXX := $(filter %$(CXXEXT),$(SRCS))
SRCS_LEX := $(filter %$(LEXEXT),$(SRCS))
SRCS_YACC := $(filter %$(YACCEXT),$(SRCS))

ifneq ($(strip $(SRCS_CXX)),)
  LDFLAGS += -lstdc++
endif

ifneq ($(strip $(SRCS_LEX)),)
  SRCS_LEX_CC := $(patsubst %$(LEXEXT),%$(LEXEXTOUT),$(SRCS_LEX))
  SRCS_LEX_HH := $(patsubst %$(CCEXT),%.h,$(SRCS_LEX_CC))
  # [solution] https://stackoverflow.com/a/46223160
  # [issue] warning: implicit declaration of function ‘fileno’
  CCFLAGS += -D_XOPEN_SOURCE=700
  LDFLAGS += -lfl
  ifeq ($(strip $(CURR_DIR)),)
    $(error ERROR$(COLON) CURR_DIR not given, for LEX $(LEXEXT))
  endif
endif

ifneq ($(strip $(SRCS_YACC)),)
  SRCS_YACC_CC := $(patsubst %$(YACCEXT),%$(YACCEXTOUT),$(SRCS_YACC))
  SRCS_YACC_HH := $(patsubst %$(CCEXT),%.h,$(SRCS_YACC_CC))
  ifeq ($(strip $(CURR_DIR)),)
    $(error ERROR$(COLON) CURR_DIR not given, for YACC $(LEXEXT))
  endif
  # INCS += $(CURR_DIR)
endif

ifneq ($(strip $(INCS)),)
  CCFLAGS += $(addprefix -I,$(patsubst %/,%,$(INCS)))
  CXXFLAGS += $(addprefix -I,$(patsubst %/,%,$(INCS)))
endif

ifneq ($(strip $(LIBS)),)
  LIB_DIRS := $(sort $(dir $(LIBS)))
  LIB_DIRS := $(patsubst %/,%,$(LIB_DIRS))
  LIB_NAMES := $(sort $(notdir $(LIBS)))
  LIB_NAMES := $(patsubst lib%.so,%,$(LIB_NAMES))

  LDFLAGS += $(addprefix -L,$(LIB_DIRS))
  LDFLAGS += $(addprefix -l,$(LIB_NAMES))
  LDFLAGS += -Wl,-rpath=$(subst $(SPACE),:,$(LIB_DIRS))
endif

SRCS_BIN := $(SRCS_CC) $(SRCS_CXX) $(SRCS_LEX_CC) $(SRCS_YACC_CC)

ifneq ($(strip $(SRCS_BIN)),)
  SRC_DIRS := $(sort $(dir $(SRCS_BIN)))
  vpath %$(CCEXT) $(subst $(SPACE),:,$(SRC_DIRS))
  vpath %$(CXXEXT) $(subst $(SPACE),:,$(SRC_DIRS))
endif

BIN := $(BINDIR)/$(TARGET)

OBJS := $(notdir $(SRCS_BIN))
OBJS := $(addsuffix .o,$(OBJS))
OBJS := $(addprefix $(OBJDIR)/,$(OBJS))

ifeq ($(NO_DEPS),)

DEPS := $(patsubst %.o,%.d,$(OBJS))
# $(info DEPS: $(DEPS))

ifeq ($(MAKECMDGOALS),)
  -include $(DEPS)
else ifeq ($(filter bin,$(MAKECMDGOALS)),bin)
  -include $(DEPS)
endif

endif

# Special Targets
#  https://www.gnu.org/software/make/manual/html_node/Special-Targets.html
.PHONY: bin print clean cleanall
.SECONDARY: $(SRCS_LEX_CC) $(SRCS_LEX_HH) $(SRCS_YACC_CC) $(SRCS_YACC_HH)

bin: $(BIN)

$(BIN): $(OBJS)
	@$(call echo,$@ < $^,1;32)
	@$(call md,$(@D),33)
	$(EXEC) $(CC) $^ $(LDFLAGS) -o $@

# LEX

%$(LEXEXTOUT): %$(LEXEXT)
	@$(call echo,$@ < $<,1;32)
	$(EXEC) $(LEX) -o $@ $<

$(OBJDIR)/%$(LEXEXTOUT).o: $(CURR_DIR)/%$(LEXEXTOUT) $(CURR_DIR)/%$(YACCEXTOUT)
	@$(call echo,$@ < $<,1;32)
	@$(call md,$(@D),33)
	$(EXEC) $(CC) $(CCFLAGS) -c $< -o $@

# .l has built-in implicit rule, override it here
%.c: %.l

# YACC

%$(YACCEXTOUT): %$(YACCEXT)
	@$(call echo,$@ < $<,1;32)
	$(EXEC) $(YACC) -d -o $@ $<

$(OBJDIR)/%$(YACCEXTOUT).o: $(CURR_DIR)/%$(YACCEXTOUT) $(CURR_DIR)/%$(LEXEXTOUT)
	@$(call echo,$@ < $<,1;32)
	@$(call md,$(@D),33)
	$(EXEC) $(CC) $(CCFLAGS) -c $< -o $@

# .y has built-in implicit rule, override it here
#  https://stackoverflow.com/a/12362834
#  https://www.gnu.org/software/make/manual/html_node/Catalogue-of-Rules.html#Catalogue-of-Rules
%.c: %.y

# CC

$(OBJDIR)/%$(CCEXT).d: %$(CCEXT)
	@$(call echo,$@ < $<,1;32)
	@$(call md,$(@D),33)
	$(EXEC) $(CC) $(CCFLAGS) -MM -MF"$@" -MT"$(@:.d=.o)" $<

$(OBJDIR)/%$(CCEXT).o: %$(CCEXT)
	@$(call echo,$@ < $<,1;32)
	@$(call md,$(@D),33)
	$(EXEC) $(CC) $(CCFLAGS) -c $< -o $@

# CXX

$(OBJDIR)/%$(CXXEXT).d: %$(CXXEXT)
	@$(call echo,$@ < $<,1;32)
	@$(call md,$(@D),33)
	$(EXEC) $(CXX) $(CXXFLAGS) -MM -MF"$@" -MT"$(@:.d=.o)" $<

$(OBJDIR)/%$(CXXEXT).o: %$(CXXEXT)
	@$(call echo,$@ < $<,1;32)
	@$(call md,$(@D),33)
	$(EXEC) $(CXX) $(CXXFLAGS) -c $< -o $@

# others

print: base_print
	@echo
	@$(call echo,Make $@)
	@echo TARGET: $(TARGET)
	@echo INCS: $(INCS)
	@echo LIBS: $(LIBS)
	@echo SRCS: $(SRCS_BIN)
	@echo OUTDIR: $(OUTDIR)
	@echo
	@echo TRTDIR: $(TRTDIR)
	@echo BINDIR: $(BINDIR)
	@echo OBJDIR: $(OBJDIR)
	@echo
	@echo BIN: $(BIN)
	@echo OBJS: $(OBJS)
	@echo DEPS: $(DEPS)

clean:
	@$(call echo,Make $@)
	@$(call rm,$(TRTDIR))
	@$(call rm_l,$(SRCS_LEX_CC))
	@$(call rm_l,$(SRCS_LEX_HH))
	@$(call rm_l,$(SRCS_YACC_CC))
	@$(call rm_l,$(SRCS_YACC_HH))

cleanall:
	@$(call echo,Make $@)
	@$(call rm,$(OUTDIR))
