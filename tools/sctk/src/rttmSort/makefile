# **********************************************************
# Makefile for rfilter1 Directory
# **********************************************************
SHELL = /bin/sh
PREFIX=/home/luodongri/code/git/eesen/tools/sctk
bindir = $(PREFIX)/bin
T=../test_suite

all: rttmSort.pl

rttmSort.pl: 
	chmod +x rttmSort.pl

check: all
	@echo "No Tests for rttmSort.pl"

install: all
	cp rttmSort.pl $(bindir)

clean:

distclean: clean
	rm -r *~ makefile
