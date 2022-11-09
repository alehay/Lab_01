test : main.c
	   gcc -fopenmp  main.c rngs.c -o test 
clean:
	rm -f test
