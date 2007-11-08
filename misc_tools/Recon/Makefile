UW2DIR = ./punwrap2D
UW3DIR = ./punwrap3D
ATLASDIR = /usr/lib/atlas

CPP = g++
CC = gcc
DEBUG = -g
CFLAGS = -Wall -O $(DEBUG)
LDFLAGS = -Wall -L$(UW2DIR) -L$(UW3DIR) -L$(ATLASDIR) -lm -lfftw3 -llapack \
		-lcblas -lunwrap2D -lunwrap3D

CLEANUPS = $(shell find ./ -maxdepth 1 -name "*.o")
CLEANUPS += $(shell find ./ -maxdepth 1 -name "*~")
CLEANALLS = $(shell find ./ -maxdepth 1 -name "recon")
CLEANALLS += $(CLEANUPS)

SRCS=bpc.c data.c geo_undist.c image.c recon.c ubpc.c util.c
OBJS=bpc.o data.o geo_undist.o image.o recon.o ubpc.o util.o

$(SRCS):
	$(CC) $(CFLAGS) -c $*.c

uwlib:
	cd $(UW2DIR); make libunwrap2D.a
	cd $(UW3DIR); make libunwrap3D.a

recon: uwlib $(OBJS)
	$(CC) -o $@ $(OBJS) $(LDFLAGS)

all: recon

clean:
ifneq ($(strip $(CLEANUPS)),)
	\rm $(CLEANUPS)
endif
	cd $(UW2DIR); make clean
	cd $(UW3DIR); make clean

cleanall:
ifneq ($(strip $(CLEANALLS)),)
	\rm $(CLEANALLS)
endif
	cd $(UW2DIR); make cleanall
	cd $(UW3DIR); make cleanall
