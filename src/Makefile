.PHONY: all perceptron_static_lib ex_1 ex_2 dvi clean


perceptron_static_lib:
	cmake -B build
	make -C build


ex_1: perceptron_static_lib
	g++ -o ex_1 experiments/ex_1.cc -Lbuild -lperceptron -lblas -std=c++17
	./ex_1

ex_2: perceptron_static_lib
	g++ -o ex_1 experiments/ex_1.cc -Lbuild -lperceptron -lblas -std=c++17
	./ex_2

dvi:
	mkdir -p documentation
	@doxygen Doxyfile && mv html documentation && mv latex documentation
	@open documentation/html/index.html

clean:
	rm -rf build
	rm -rf documentation
	rm -rf ex_*


