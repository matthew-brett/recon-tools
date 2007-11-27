CC = gcc
DEBUG = -g
CFLAGS = -Wall -O $(DEBUG)
UNAME = $(shell uname)
ifeq "$(UNAME)" "Darwin"
	CFLAGS += -DDARWIN
endif
AR=ar
RANLIB=ranlib

CLEANUPS = $(shell find . -maxdepth 1 -name "*.o")
CLEANUPS += $(shell find . -maxdepth 1 -name "*~")
CLEANALLS = $(CLEANUPS) $(shell find . -maxdepth 1 -name "libunwrap2D.a")
OBJ=Munther_2D_unwrap.o

all: libunwrap2D.a
	
$(OBJ): 
	$(CC) $(CFLAGS) -c $*.c

libunwrap2D.a: $(OBJ)
	$(AR) -rvu $@ $(OBJ)
	$(RANLIB) $@

clean:
ifneq ($(strip $(CLEANUPS)),)
	\rm $(CLEANUPS)
endif

cleanall:
ifneq ($(strip $(CLEANALLS)),)
	\rm $(CLEANALLS)
endif
