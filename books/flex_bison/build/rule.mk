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
# $(info SRCS_CC: $(SRCS_CC))
# $(info SRCS_CXX: $(SRCS_CXX))
# $(info SRCS_LEX: $(SRCS_LEX))

ifneq ($(strip $(INCS)),)
  CCFLAGS += $(addprefix -I,$(patsubst %/,%,$(INCS)))
  CXXFLAGS += $(addprefix -I,$(patsubst %/,%,$(INCS)))
endif

ifneq ($(strip $(SRCS_CXX)),)
  LDFLAGS += -lstdc++
endif

ifneq ($(strip $(SRCS_LEX)),)
  SRCS_LEX_CC := $(patsubst %$(LEXEXT),%$(LEXEXTOUT),$(SRCS_LEX))
  CCFLAGS += -Wno-implicit-function-declaration -Wno-unused-function
  LDFLAGS += -lfl
  ifeq ($(strip $(CURR_DIR)),)
    $(error ERROR$(COLON) CURR_DIR not given, for LEX $(LEXEXT))
  endif
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

SRCS_BIN := $(SRCS_CC) $(SRCS_CXX) $(SRCS_LEX_CC)

ifneq ($(strip $(SRCS_BIN)),)
  SRC_DIRS := $(sort $(dir $(SRCS_BIN)))
  vpath %$(CCEXT) $(subst $(SPACE),:,$(SRC_DIRS))
  vpath %$(CXXEXT) $(subst $(SPACE),:,$(SRC_DIRS))
endif

BIN := $(BINDIR)/$(TARGET)

OBJS := $(notdir $(SRCS_BIN))
OBJS := $(addsuffix .o,$(OBJS))
OBJS := $(addprefix $(OBJDIR)/,$(OBJS))

DEPS := $(patsubst %.o,%.d,$(OBJS))
# $(info DEPS: $(DEPS))

ifeq ($(MAKECMDGOALS),)
  -include $(DEPS)
else ifeq ($(filter bin,$(MAKECMDGOALS)),bin)
  -include $(DEPS)
endif

# Special Targets
#  https://www.gnu.org/software/make/manual/html_node/Special-Targets.html
.PHONY: bin print clean cleanall
.SECONDARY: $(SRCS_LEX_CC)

bin: $(BIN)

$(BIN): $(OBJS)
	@$(call echo,$@ < $^,1;32)
	@$(call md,$(@D),33)
	$(EXEC) $(CC) $^ $(LDFLAGS) -o $@

# LEX

%$(LEXEXTOUT): %$(LEXEXT)
	@$(call echo,$@ < $<,1;32)
	$(EXEC) $(LEX) -o $@ $<

$(OBJDIR)/%$(LEXEXTOUT).o: $(CURR_DIR)/%$(LEXEXTOUT)
	@$(call echo,$@ < $<,1;32)
	@$(call md,$(@D),33)
	$(EXEC) $(CC) $(CCFLAGS) -c $< -o $@

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

cleanall:
	@$(call echo,Make $@)
	@$(call rm,$(OUTDIR))