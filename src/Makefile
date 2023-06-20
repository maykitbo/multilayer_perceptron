# -lOpenCL -lclBLAS

all: clean
	g++ main.cc lib/m_thread.cc common/metrics.cc -lpthread -lm -ldl -lblas -Ofast -march=native
	./a.out

qt:
	cmake -B"build"
	$(MAKE) -C build
	mv build/mlp mlp
	./mlp

clean:
	rm -rf a.out
	rm -rf build/*
	rm -rf mlp
	rm -rf ../build-*
