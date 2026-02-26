# LearnCUDA — Root Makefile
# Builds all chapters that contain CUDA C code (Chapters 01-10, 12).
# Chapter 11 (Python) is built separately via pip/setuptools.

CHAPTERS = Chapter_01 Chapter_02 Chapter_03 Chapter_04 Chapter_05 \
           Chapter_06 Chapter_07 Chapter_08 Chapter_09 Chapter_10 Chapter_12

.PHONY: all clean $(CHAPTERS)

all: $(CHAPTERS)

$(CHAPTERS):
	@echo "=== Building $@ ==="
	$(MAKE) -C $@
	@echo ""

clean:
	@for ch in $(CHAPTERS); do \
		echo "Cleaning $$ch ..."; \
		$(MAKE) -C $$ch clean; \
	done

# Build individual chapters
ch01: Chapter_01
ch02: Chapter_02
ch03: Chapter_03
ch04: Chapter_04
ch05: Chapter_05
ch06: Chapter_06
ch07: Chapter_07
ch08: Chapter_08
ch09: Chapter_09
ch10: Chapter_10
ch12: Chapter_12
