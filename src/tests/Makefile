.PHONY: file_test clean speed_test settings_test learn_test valgrind

CC=g++
FLAGS=-Wall -Werror -Wextra -std=c++17
TESTFLAGS=-lgtest_main -lgtest -pthread
SRC=../common/metrics.cc ../common/perceptron_settings.cc ../common/error_thread.cc ../common/functions.cc\
		../model/data_manager.cc ../model/model.cc ../model/matrix_model.cc ../model/graph_model.cc
MM_SRC=../model/matrix_model.cc
GM_SRC=../model/graph_model.cc
OPTIMIZATION=-Ofast -march=native
VALGRIND_FLAGS=--track-origins=yes --trace-children=yes --leak-check=full --leak-resolution=med

all: file_test settings_test learn_test


file_test:
	$(CC) $(FLAGS) $(SRC) file.cc $(TESTFLAGS) -o file_test $(OPTIMIZATION)
	./file_test

speed_test:
	$(CC) $(FLAGS) $(SRC) speed.cc -o speed_test $(OPTIMIZATION)
	./speed_test

settings_test:
	$(CC) $(FLAGS) $(SRC) settings.cc $(TESTFLAGS) -o settings_test $(OPTIMIZATION)
	./settings_test

learn_test:
	$(CC) $(FLAGS) $(SRC) learn.cc $(TESTFLAGS) -o learn_test $(OPTIMIZATION)
	./learn_test

valgrind:
	$(CC) $(FLAGS) $(SRC) file.cc $(TESTFLAGS) -o file_test
	valgrind --log-file="file_test.txt" $(VALGRIND_FLAGS)  ./file_test
	! grep -n "ERROR SUMMARY" file_test.txt | grep -v "0 errors"
	$(CC) $(FLAGS) $(SRC) settings.cc $(TESTFLAGS) -o settings_test
	valgrind --log-file="settings_test.txt" $(VALGRIND_FLAGS) ./settings_test
	! grep -n "ERROR SUMMARY" settings_test.txt | grep -v "0 errors"
	$(CC) $(FLAGS) $(SRC) learn.cc $(TESTFLAGS) -o learn_test
	valgrind --log-file="learn_test.txt" $(VALGRIND_FLAGS) ./learn_test
	! grep -n "ERROR SUMMARY" learn_test.txt | grep -v "0 errors"

clean:
	rm -rf file_test
	rm -rf speed_test
	rm -rf settings_test
	rm -rf learn_test
	rm -rf *.txt