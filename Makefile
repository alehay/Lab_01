test : main.c
	   gcc -Wincompatible-pointer-types -fopenmp  main.c rngs.c -o test 
clean:
	rm -f test
