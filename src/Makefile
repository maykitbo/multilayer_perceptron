.PHONY: all perceptron_static_lib ex_1 ex_2 dvi clean letters_visuzalization graph


perceptron_static_lib:
	cmake -B build
	make -C build

ex_1: perceptron_static_lib
	g++ -o ex_1 experiments/ex_1.cc -Lbuild -lperceptron -lblas -std=c++17
	./ex_1

ex_2: perceptron_static_lib
	g++ -o ex_2 experiments/ex_2.cc -Lbuild -lperceptron -lblas -std=c++17
	./ex_2

letters_visuzalization: perceptron_static_lib
	cmake experiments/ -B lv_build
	make -C lv_build
	./lv_build/letters_visualization

graph: perceptron_static_lib
	cmake experiments/graph -B g_build
	make -C g_build
	./g_build/graph_g

dvi:
	mkdir -p documentation
	@doxygen Doxyfile && mv html documentation && mv latex documentation
	@open documentation/html/index.html

clean:
	rm -rf build
	rm -rf documentation
	rm -rf ex_*
	rm -rf lv_build
	rm -rf g_build


