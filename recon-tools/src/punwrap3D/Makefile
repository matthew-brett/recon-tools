CC = gcc
DEBUG = -g
CFLAGS = -Wall -O $(DEBUG)
AR=ar
RANLIB=ranlib

CLEANUPS = $(shell find ./ -maxdepth 1 -name "*.o")
CLEANUPS += $(shell find ./ -maxdepth 1 -name "*~")
CLEANALLS = $(CLEANUPS) $(shell find ./ -maxdepth 1 -name "libunwrap3D.a")

SRCS=$(shell find . -maxdepth 1 -name "*.c")
OBJS=$(SRCS:.c=.o)

all: libunwrap3D.a

foo:
	@echo $(SRCS)
	@echo $(OBJS)

libunwrap3D.a: $(OBJS)
	$(AR) -rvu $@ $(OBJS)
	$(RANLIB) $@

clean:
	@echo $(CLEANUPS)
ifneq ($(strip $(CLEANUPS)),)
	\rm $(CLEANUPS)
endif

cleanall:
ifneq ($(strip $(CLEANALLS)),)
	\rm $(CLEANALLS)
endif